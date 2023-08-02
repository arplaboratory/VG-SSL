import torch
import torch.nn as nn
import torch.nn.functional as F

from model.byol.byol_pytorch import NetWrapper, EMA, get_module_device, set_requires_grad, update_moving_average
from model.vicreg.vicreg_pytorch import FullGatherLayer
import logging
import copy

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def info_nce_loss(features, device, batch_size, temperature, n_views=2):

    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * batch_size, self.args.n_views * batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits.div(temperature)
    return logits, labels

class MOCO(nn.Module):
    """
    Build a MoCov2 model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self,
                 net,
                 image_size,
                 num_nodes,
                 num_devices,
                 projection_size = 128,
                 projection_hidden_size = 2048,
                 hidden_layer = -1,
                 K=65536,
                 moving_average_decay = 0.999,
                 T=0.07,
                 shuffle_bn=False,
                 use_simclr=False,
                 aggregation=None,
                 n_layers=2):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        self.net = net
        self.aggregation = aggregation
        self.criterion = nn.CrossEntropyLoss()
        self.num_nodes = num_nodes
        self.num_devices = num_devices
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        # Augmentation is finished outside

        self.K = K
        self.T = T
        self.shuffle_bn = shuffle_bn
        self.use_simclr = use_simclr

        # create the encoders
        # num_classes is the output fc dimension
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer, mlp="NoBnMLP", aggregation=self.aggregation, n_layers=n_layers)
        self.use_momentum = True
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        # get device of network and make wrapper same device
        self.device = get_module_device(self.net)
        self.to(self.device)

        # create the queue
        if not self.use_simclr:
            self.register_buffer("queue", torch.randn(projection_size, K))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # send a mock image tensor to instantiate singleton parameters
        self.eval()
        self.forward(torch.randn(2, 3, image_size[0], image_size[1], device=self.device), torch.randn(2, 3, image_size[0], image_size[1], device=self.device))
        self.train()

    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if not(self.num_devices==1 and self.num_nodes==1):
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0] # Use local batch size here

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, return_embedding=False, return_projection=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        if return_embedding:
            return self.online_encoder(im_q, return_projection = return_projection)

        # compute query features
        q = self.online_encoder(im_q, return_projection = False)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        if self.use_simclr:
            if self.target_encoder is None:
                self.target_encoder = self.online_encoder
                return

            k = self.target_encoder(im_k, return_projection = False)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
        else:
            with torch.no_grad():  # no gradient to keys
                if self.target_encoder is None:
                    self.target_encoder = self._get_target_encoder()
                    return

                # Momentum Update is done before zero_grad() outside

                if self.shuffle_bn:
                    # shuffle for making use of BN
                    im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                k = self.target_encoder(im_k, return_projection = False)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)

                if self.shuffle_bn:
                    # undo shuffle
                    k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        if self.use_simclr:
            # simclr logit and label calculation
            # feature should be concated as 
            # query_1 query_2 query_3 ... query_n pos_1 pos2 pos3 ... pos_n
            # n is batch size
            if not (self.num_devices==1 and self.num_nodes==1):
                q = torch.cat(FullGatherLayer.apply(q), dim=0)
                k = torch.cat(FullGatherLayer.apply(k), dim=0)
            features = torch.cat([q, k], dim = 0)
            batch_size = im_q.shape[0] * self.num_nodes * self.num_devices # Infer global batch size here
            logits, labels = info_nce_loss(features, self.device, batch_size, self.T)
        else:
            # moco logit and label calculation
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1).to(self.device)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # dequeue and enqueue
        if not self.use_simclr:
            self._dequeue_and_enqueue(k)

        loss = self.criterion(logits, labels)

        return loss.mean()
