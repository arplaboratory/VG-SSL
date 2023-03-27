import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T
import torch.distributed as dist
from model.byol_pytorch import NetWrapper

@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


def cluster_memory(model, local_memory_index, local_memory_embeddings, size_dataset, nmb_kmeans_iters=10):
    j = 0
    assignments = -100 * torch.ones(len(args.nmb_prototypes), size_dataset).long()
    with torch.no_grad():
        for i_K, K in enumerate(args.nmb_prototypes):
            # run distributed k-means

            # init centroids with elements from memory bank of rank 0
            centroids = torch.empty(K, args.feat_dim).cuda(non_blocking=True)
            if args.rank == 0:
                random_idx = torch.randperm(len(local_memory_embeddings[j]))[:K]
                assert len(random_idx) >= K, "please reduce the number of centroids"
                centroids = local_memory_embeddings[j][random_idx]
            dist.broadcast(centroids, 0)

            for n_iter in range(nmb_kmeans_iters + 1):

                # E step
                dot_products = torch.mm(local_memory_embeddings[j], centroids.t())
                _, local_assignments = dot_products.max(dim=1)

                # finish
                if n_iter == nmb_kmeans_iters:
                    break

                # M step
                where_helper = get_indices_sparse(local_assignments.cpu().numpy())
                counts = torch.zeros(K).cuda(non_blocking=True).int()
                emb_sums = torch.zeros(K, args.feat_dim).cuda(non_blocking=True)
                for k in range(len(where_helper)):
                    if len(where_helper[k][0]) > 0:
                        emb_sums[k] = torch.sum(
                            local_memory_embeddings[j][where_helper[k][0]],
                            dim=0,
                        )
                        counts[k] = len(where_helper[k][0])
                dist.all_reduce(counts)
                mask = counts > 0
                dist.all_reduce(emb_sums)
                centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

                # normalize centroids
                centroids = nn.functional.normalize(centroids, dim=1, p=2)

            getattr(model.module.prototypes, "prototypes" + str(i_K)).weight.copy_(centroids)

            # gather the assignments
            assignments_all = torch.empty(args.world_size, local_assignments.size(0),
                                          dtype=local_assignments.dtype, device=local_assignments.device)
            assignments_all = list(assignments_all.unbind(0))
            dist_process = dist.all_gather(assignments_all, local_assignments, async_op=True)
            dist_process.wait()
            assignments_all = torch.cat(assignments_all).cpu()

            # gather the indexes
            indexes_all = torch.empty(args.world_size, local_memory_index.size(0),
                                      dtype=local_memory_index.dtype, device=local_memory_index.device)
            indexes_all = list(indexes_all.unbind(0))
            dist_process = dist.all_gather(indexes_all, local_memory_index, async_op=True)
            dist_process.wait()
            indexes_all = torch.cat(indexes_all).cpu()

            # log assignments
            assignments[i_K][indexes_all] = assignments_all

            # next memory bank to use
            j = (j + 1) % len(args.crops_for_assign)

    return assignments

# main class

class SWAV(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        prototype = None,
        aggregation = None,
    ):
        super().__init__()
        self.net = net
        self.aggregation = aggregation
        # Augmentation is finished outside

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer, use_simsiam_mlp=not use_momentum, aggregation=self.aggregation)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size[0], image_size[1], device=device), torch.randn(2, 3, image_size[0], image_size[1], device=device))

    def forward(
        self,
        x,
        y,
        return_embedding = False,
        return_projection = True
    ):
        # assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)

        # image one and two are proceeded outsides
        image_one, image_two = x, y

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        with torch.no_grad():
            if self.target_encoder is None:
                self.target_encoder = self._get_target_encoder()
            target_encoder =  self.target_encoder if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()