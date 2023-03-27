
import os
import torch
import logging
import torchvision
from torch import nn
from os.path import join
from transformers import ViTModel
from google_drive_downloader import GoogleDriveDownloader as gdd
from transformers import ViTMAEConfig, ViTMAEModel
import torch.nn.functional as F

import torch.distributed as dist


from model.cct import cct_14_7x2_384
from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation
from model.non_local import NonLocalBlock
from model.byol.byol_pytorch import BYOL
from model.sync_batchnorm import convert_model
from model.vicreg.utils import adjust_learning_rate, LARS, exclude_bias_and_norm
import datasets_ws
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import numpy as np
import faiss

sim_coeff = 25.0
std_coeff = 25.0
cov_coeff = 1.0

num_features = 2048

# Pretrained models on Google Landmarks v2 and Places 365
PRETRAINED_MODELS = {
    'resnet18_places'  : '1DnEQXhmPxtBUrRc81nAvT8z17bk-GBj5',
    'resnet50_places'  : '1zsY4mN4jJ-AsmV3h4hjbT72CBfJsgSGC',
    'resnet101_places' : '1E1ibXQcg7qkmmmyYgmwMTh7Xf1cDNQXa',
    'vgg16_places'     : '1UWl1uz6rZ6Nqmp1K5z3GHAIZJmDh4bDu',
    'resnet18_gldv2'   : '1wkUeUXFXuPHuEvGTXVpuP5BMB-JJ1xke',
    'resnet50_gldv2'   : '1UDUv6mszlXNC1lv6McLdeBNMq9-kaA70',
    'resnet101_gldv2'  : '1apiRxMJpDlV0XmKlC5Na_Drg2jtGL-uE',
    'vgg16_gldv2'      : '10Ov9JdO7gbyz6mB5x0v_VSAUMj91Ta4o'
}

PRETRAINED_SSL_MODELS = {
    'simclr' : 'simclr-resnet50',
    'byol': 'byol-resnet50',
    'vicreg': 'vicreg-resnet50',
    'swav': 'swav-resnet50',
    'bt': 'bt-resnet50',
    'moco': 'moco-resnet50',
    'mocov2': 'mocov2-resnet50',
    'simsiam': 'simsiam-resnet50'
}

class VicregVGNet(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.arch_name = args.backbone
        self.aggregation = get_aggregation(args)
   
        self.projector = Projector(args, 65536)
        self.batch_size = args.train_batch_size

    

        if args.fc_output_dim != None:
        # Concatenate fully connected layer to the aggregation layer
            self.aggregation = nn.Sequential(self.aggregation,
                                                nn.Linear(args.features_dim, args.fc_output_dim),
                                                L2Norm())
            args.features_dim = args.fc_output_dim
        

    def forward(self, x, y):

        x = self.backbone(x)
 
        y = self.backbone(y)
        
        x = self.aggregation(x)
        y = self.aggregation(y)
     

        x = self.projector(x)
        y = self.projector(y)
        
        repr_loss = F.mse_loss(x, y)

        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(num_features)

        loss = (
            sim_coeff * repr_loss
            + std_coeff * std_loss
            + cov_coeff * cov_loss
        )
        return loss

def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.arch_name = args.backbone
        self.aggregation = get_aggregation(args)
        self.self_att = False

        #self.projector = Projector(args, 16384)
        


        if args.aggregation in ["gem", "spoc", "mac", "rmac"]:
            if args.l2 == "before_pool":
                self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())
            elif args.l2 == "after_pool":
                self.aggregation = nn.Sequential(self.aggregation, L2Norm(), Flatten())
            elif args.l2 == "none":
                self.aggregation = nn.Sequential(self.aggregation, Flatten())
        
        if args.fc_output_dim != None:
            # Concatenate fully connected layer to the aggregation layer
            self.aggregation = nn.Sequential(self.aggregation,
                                             nn.Linear(args.features_dim, args.fc_output_dim),
                                             L2Norm())
            args.features_dim = args.fc_output_dim
        if args.non_local:
            non_local_list = [NonLocalBlock(channel_feat=get_output_channels_dim(self.backbone),
                                           channel_inner=args.channel_bottleneck)]* args.num_non_local
            self.non_local = nn.Sequential(*non_local_list)
            self.self_att = True

    def forward(self, x):

        
        x = self.backbone(x)
        if self.self_att:
            x = self.non_local(x)
        if self.arch_name.startswith("vit") :
            x = x.last_hidden_state[:, 0, :]
            return x
        x = self.aggregation(x)
        #x = self.projector(x)
        
        return x


def Projector(args, embedding):
    mlp = "2048-2048"
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

def get_aggregation(args):
    if args.aggregation == "gem":
        return aggregation.GeM(work_with_tokens=args.work_with_tokens)
    elif args.aggregation == "spoc":
        return aggregation.SPoC()
    elif args.aggregation == "mac":
        return aggregation.MAC()
    elif args.aggregation == "rmac":
        return aggregation.RMAC()
    elif args.aggregation == "netvlad":
        return aggregation.NetVLAD(clusters_num=args.netvlad_clusters, dim=args.features_dim,
                                   work_with_tokens=args.work_with_tokens)
    elif args.aggregation == 'crn':
        return aggregation.CRN(clusters_num=args.netvlad_clusters, dim=args.features_dim)
    elif args.aggregation == "rrm":
        return aggregation.RRM(args.features_dim)
    elif args.aggregation == 'none'\
            or args.aggregation == 'cls' \
            or args.aggregation == 'seqpool':
        return nn.Identity()


def get_pretrained_model(args):
    if args.pretrain == 'places':  num_classes = 365
    elif args.pretrain == 'gldv2':  num_classes = 512
    elif args.pretrain in PRETRAINED_SSL_MODELS: num_classes = 1000
    else:
        raise NotImplementedError()
    
    # Check SSL model backbone
    if args.pretrain in PRETRAINED_SSL_MODELS:
        assert(args.backbone.startswith("resnet50"))

    if args.backbone.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif args.backbone.startswith("resnet50"):
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.backbone.startswith("resnet101"):
        model = torchvision.models.resnet101(num_classes=num_classes)
    elif args.backbone.startswith("vgg16"):
        model = torchvision.models.vgg16(num_classes=num_classes)
    
    if args.backbone.startswith('resnet'):
        model_name = args.backbone.split('conv')[0] + "_" + args.pretrain
    else:
        model_name = args.backbone + "_" + args.pretrain

    if args.pretrain in PRETRAINED_SSL_MODELS:
        file_path = join("pretrained", PRETRAINED_SSL_MODELS[args.pretrain] + ".pth")
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))
        if args.pretrain == 'simclr':
            state_dict = state_dict['state_dict']
        elif args.pretrain == 'byol' or args.pretrain == 'vicreg' or args.pretrain == 'bt':
            state_dict = state_dict
        elif args.pretrain == 'swav':
            update_state_dict = dict()
            for key, value in state_dict.items():
                remove_prefix_key = key.replace('module.', '')
                update_state_dict[remove_prefix_key] = value
            state_dict = update_state_dict
        elif args.pretrain == 'moco' or args.pretrain == 'mocov2':
            update_state_dict = dict()
            for key, value in state_dict.items():
                remove_prefix_key = key.replace('module.encoder_q.', '')
                update_state_dict[remove_prefix_key] = value
            state_dict = update_state_dict
        elif args.pretrain == 'simsiam':
            state_dict = state_dict['state_dict']
            update_state_dict = dict()
            for key, value in state_dict.items():
                remove_prefix_key = key.replace('module.encoder.', '')
                update_state_dict[remove_prefix_key] = value
            state_dict = update_state_dict
        else:
            raise NotImplementedError()
        model.load_state_dict(state_dict, strict=False)
    else:
        file_path = join("data", "pretrained_nets", model_name +".pth")
        if not os.path.exists(file_path):
            gdd.download_file_from_google_drive(file_id=PRETRAINED_MODELS[model_name],
                                                dest_path=file_path)
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
            update_state_dict = dict()
            for key, value in state_dict.items():
                remove_prefix_key = key.replace('module.encoder.', '')
                update_state_dict[remove_prefix_key] = value
            update_state_dict.pop('fc.weight', None)
            update_state_dict.pop('fc.bias', None)
            state_dict = update_state_dict
        model.load_state_dict(state_dict, strict=False)
    return model


def get_backbone(args):
    # The aggregation layer works differently based on the type of architecture
    args.work_with_tokens = args.backbone.startswith('cct') or args.backbone.startswith('vit')
    if args.backbone.startswith("resnet"):
        if args.pretrain in ['places', 'gldv2', 'simclr', 'byol', 'vicreg', 'swav', 'bt', 'moco', 'mocov2', 'simsiam']:
            backbone = get_pretrained_model(args)
        elif args.backbone.startswith("resnet18"):
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone.startswith("resnet50"):
            backbone = torchvision.models.resnet50(pretrained=True)
        elif args.backbone.startswith("resnet101"):
            backbone = torchvision.models.resnet101(pretrained=True)
        if not args.unfreeze:
            for name, child in backbone.named_children():
                # Freeze layers before conv_3
                if name == "layer3":
                    break
                for params in child.parameters():
                    params.requires_grad = False
        if args.backbone.endswith("conv4"):
            if not args.unfreeze:
                logging.debug(f"Train only conv4_x of the {args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
            else:
                logging.debug(f"Train only conv4_x of the {args.backbone.split('conv')[0]} (remove conv5_x)")
            layers = list(backbone.children())[:-3]
        elif args.backbone.endswith("conv5"):
            if not args.unfreeze:
                logging.debug(f"Train only conv4_x and conv5_x of the {args.backbone.split('conv')[0]}, freeze the previous ones")
            else:
                logging.debug(f"Train only conv4_x and conv5_x of the {args.backbone.split('conv')[0]}")
            layers = list(backbone.children())[:-2]
    elif args.backbone == "vgg16":
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        else:
            backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        if not args.unfreeze:
            logging.debug("Train last layers of the vgg16, freeze the previous ones")
        else:
            logging.debug("Train last layers of the vgg16")
    elif args.backbone == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:5]:
            for p in l.parameters(): p.requires_grad = False
        if not args.unfreeze:
            logging.debug("Train last layers of the alexnet, freeze the previous ones")
        else:
            logging.debug("Train last layers of the alexnet")
    elif args.backbone.startswith("cct"):
        if args.backbone.startswith("cct384"):
            backbone = cct_14_7x2_384(pretrained=True, progress=True, aggregation=args.aggregation)
        if args.trunc_te:
            logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
            backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:args.trunc_te].children())
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.classifier.blocks.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 384
        return backbone
    elif args.backbone == ("vit"):
        if args.resize[0] == 224:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        elif args.resize[0] == 384:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')
        else:
            raise ValueError('Image size for ViT must be either 224 or 384')

        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te+1}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 768
        return backbone
    elif args.backbone == ("vitmae"):
        if args.resize[0] == 224:
            backbone = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        else:
            raise ValueError('Image size for ViT must be either 224 or 384')
        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te+1}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 768
        return backbone

    
    
    backbone = torch.nn.Sequential(*layers)
    args.features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    return backbone

def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]

def setup_optimizer_loss(args, model_parameters, return_loss=True):
    # Setup Optimizer
    if args.aggregation == "crn":
        raise NotImplementedError()
    else:
        if args.optim == "adam":
            if args.cosine_scheduler:
                optimizer = torch.optim.Adam(model_parameters, lr=0)
            else:
                optimizer = torch.optim.Adam(model_parameters, lr=args.lr)
        elif args.optim == "sgd":
            if args.cosine_scheduler:
                optimizer = torch.optim.SGD(
                    model_parameters, lr=0, momentum=0.9, weight_decay=1e-6
                )
            else:
                optimizer = torch.optim.SGD(
                    model_parameters, lr=args.lr, momentum=0.9, weight_decay=1e-6
                )
        elif args.optim == "lars":
            if args.cosine_scheduler:
                optimizer = LARS(
                    model_parameters, lr=0, weight_decay=1e-6, weight_decay_filter=exclude_bias_and_norm,
                    lars_adaptation_filter=exclude_bias_and_norm
                    )
            else:
                optimizer = LARS(
                    model_parameters, lr=args.lr, weight_decay=1e-6, weight_decay_filter=exclude_bias_and_norm,
                    lars_adaptation_filter=exclude_bias_and_norm
                )
    # Setup Loss
    if args.method == "pair":
        # TODO: Add pair loss criterion here. If the model return loss, then skip
        if return_loss == False:
            raise NotImplementedError("Criterion not found for pairs!")
        else:
            criterion_pairs = None
    else:
        raise NotImplementedError()
    return optimizer, criterion_pairs

class SSLGeoLocalizationNet(pl.LightningModule):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args, ds_list):
        super().__init__()
        self.args = args
        self.backbone = get_backbone(args)
        self.aggregation = get_aggregation(args)
        self.arch_name = args.backbone
        self.return_loss = False
        self.ssl_model = self.get_ssl_model()
        self.train_ds, self.val_ds, self.test_ds = ds_list
        self.all_features = None
        self.best_r5 = None
        self.lr = 0

    def get_ssl_model(self):
        if self.args.ssl_method == "byol":
            self.return_loss = True
            return BYOL(self.backbone,
                        hidden_layer = -1,
                        image_size=self.args.resize,
                        aggregation = self.aggregation)
        if self.args.ssl_method == "simsiam":
            self.return_loss = True
            return BYOL(self.backbone,
                        hidden_layer = -1,
                        image_size = self.args.resize,
                        aggregation = self.aggregation,
                        use_momentum=False)
        else:
            raise NotImplementedError()

    def forward(self, x, pairs_local_indexes=None, return_feature=False):
        if return_feature:
            if self.args.ssl_method == "byol" or self.args.ssl_method == "simsiam":
                feature = self.ssl_model(x, y=None, return_embedding=True, return_projection=False)
            else:
                raise NotImplementedError()
            return feature
        if pairs_local_indexes is None:
            raise NotImplementedError("pairs indexes should be pass if not return_feature")
        x_indexes = pairs_local_indexes[0:len(pairs_local_indexes):2].long()
        y_indexes = pairs_local_indexes[1:len(pairs_local_indexes):2].long()
        input_x = x[x_indexes]
        input_y = x[y_indexes]
        if self.return_loss:
            loss = self.ssl_model(input_x, input_y).mean()
            return loss
        else:
            raise NotImplementedError()
        
    def update(self):
        if self.args.ssl_method == "byol":
            self.ssl_model.update_moving_average()
        elif self.args.ssl_method == "simsiam":
            pass
        else:
            raise NotImplementedError()
    
    def train_dataloader(self):
        if self.args.method == 'pair':
            # Compute pairs to use in the pair loss
            self.train_ds.is_inference = True
            self.train_ds.compute_pairs(self.args, None)
            self.train_ds.is_inference = False
            pairs_dl = DataLoader(
                dataset=self.train_ds,
                num_workers=self.args.num_workers,
                batch_size=self.args.train_batch_size,
                collate_fn=datasets_ws.collate_fn
            )
            return pairs_dl
        else:
            raise NotImplementedError()

    def val_dataloader(self):
        val_dataloader = DataLoader(
            dataset=self.val_ds,
            num_workers=self.args.num_workers,
            batch_size=self.args.infer_batch_size,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            dataset=self.test_ds,
            num_workers=self.args.num_workers,
            batch_size=self.args.infer_batch_size,
        )
        return test_dataloader

    def training_step(self, inputs, _):
        if self.args.method == "pair":
            images, pairs_local_indexes, _ = inputs
            # Flip all pairs or none
            if self.args.horizontal_flip:
                images = transforms.RandomHorizontalFlip()(images)

            # Compute features of all images (images contains queries, positives and negatives)
            if self.criterion_pairs is None:
                loss = self.forward(images, pairs_local_indexes)
                loss_pairs = loss
            else:
                raise NotImplementedError('Unknown loss is used')
                # features = model(images.to(args.device))
                # loss_pairs = 0
                # del features

            # loss_pairs /= self.args.train_batch_size
            self.log("loss", loss_pairs, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.train_batch_size, rank_zero_only=True)
            self.log("current_lr", self.lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.train_batch_size, rank_zero_only=True)
            return {'loss': loss_pairs}
        else:
            raise NotImplementedError()
    
    def _shared_on_eval_epoch_start(self, eval_ds, args):
        if self.trainer.is_global_zero:
            self.all_features = torch.empty((len(eval_ds), self.args.features_dim), dtype=torch.float32, device="cpu")

    def _shared_eval_step(self, eval_ds, inputs):
        images, indices = inputs
        features = self.forward(images, return_feature=True)
        features = self.all_gather(features)
        indices = self.all_gather(indices)
        if self.trainer.is_global_zero:
            features = features.view(-1, features.size(-1))
            indices = indices.view(-1)
            self.all_features[indices.cpu(), :] = features.cpu()

    def _shared_on_eval_epoch_end(self, eval_ds, args):
        queries_features = self.all_features[eval_ds.database_num:]
        database_features = self.all_features[: eval_ds.database_num]

        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(database_features)
        del database_features, self.all_features
        self.all_features = None

        logging.debug("Calculating recalls")
        distances, predictions = faiss_index.search(
            queries_features, max(args.recall_values)
        )
        # For each query, check if the predictions are correct
        positives_per_query = eval_ds.get_positives()
        # args.recall_values by default is [1, 5, 10, 20]
        recalls = np.zeros(len(args.recall_values))
        for query_index, pred in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break
        # Divide by the number of queries*100, so the recalls are in percentages
        recalls = recalls / eval_ds.queries_num * 100
        recalls_str = ", ".join(
            [f"R@{val}: {rec:.1f}" for val,
                rec in zip(args.recall_values, recalls)]
        )

        return recalls, recalls_str

    def on_validation_epoch_start(self):
        self._shared_on_eval_epoch_start(self.val_ds, self.args)

    def validation_step(self, inputs, _):
        self._shared_eval_step(self.val_ds, inputs)

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            recalls, recalls_str = self._shared_on_eval_epoch_end(self.val_ds, self.args)
            if self.best_r5 is None or self.best_r5 < recalls[1]:
                self.best_r5 = recalls[1]
            self.log("val_recall1", recalls[0])
            self.log("val_recall5", recalls[1])
            self.log("val_best_r5", self.best_r5)

    def on_test_epoch_start(self):
        self._shared_on_eval_epoch_start(self.test_ds, self.args)

    def test_step(self, inputs, _):
        self._shared_eval_step(self.test_ds, inputs)

    def on_test_epoch_end(self):
        if self.trainer.is_global_zero:
            recalls, recalls_str = self._shared_on_eval_epoch_end(self.test_ds, self.args)
            log_dict = {"test_recall1":recalls[0], "test_recall5":recalls[1]}
            self.log("test_recall1", recalls[0])
            self.log("test_recall5", recalls[1])

    def configure_optimizers(self):
        optimizer, self.criterion_pairs = setup_optimizer_loss(self.args, self.ssl_model.parameters(), return_loss=True)
        return optimizer

    def on_before_zero_grad(self, _):
        if self.args.cosine_scheduler:
            self.lr = adjust_learning_rate(self.args, self.optimizers(), self.global_step, self.args.num_nodes, self.args.num_devices)
        self.update()

    def setup(self, stage):
        if stage == "validate":
            if self.args.aggregation in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
                self.ssl_model.to(self.args.device)
                if not self.args.resume:
                    self.train_ds.is_inference = True
                    self.aggregation.initialize_netvlad_layer(
                        self.args, self.train_ds, self.backbone)
                self.args.features_dim *= self.args.netvlad_clusters

    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure,
        ):
        optimizer_closure()
        optimizer.step()