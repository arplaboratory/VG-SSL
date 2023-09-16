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

from model.cct import cct_14_7x2_384
from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation
from model.non_local import NonLocalBlock
from model.byol.byol_pytorch import BYOL
from model.vicreg.vicreg_pytorch import VICREG
from model.moco.moco_pytorch import MOCO
from model.sync_batchnorm import convert_model
from model.vicreg.utils import adjust_learning_rate
import datasets_ws
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from datasets_ws import inv_base_transforms
from torchvision import transforms
import numpy as np
import faiss
from model.r2former import R2Former, AnySizePatchEmbed
from functools import partial
from model.Deit import deit_small_distilled_patch16_224, deit_base_distilled_patch16_384

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


class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.arch_name = args.backbone
        self.aggregation = get_aggregation(args)
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
        if args.n_layers != 0:
             self.aggregation = attach_projection_layers(args, self.backbone, self.aggregation, args.resize, args.projection_size)
             
        self.self_att = False
        if args.non_local:
            non_local_list = [NonLocalBlock(channel_feat=get_output_channels_dim(self.backbone),
                                           channel_inner=args.channel_bottleneck)]* args.num_non_local
            self.non_local = nn.Sequential(*non_local_list)
            self.self_att = True
        self.single = True

    def forward(self, x):   
        x = self.backbone(x)
        if self.self_att:
            x = self.non_local(x)
        if self.arch_name.startswith("vit"):
            x = x.last_hidden_state[:, 0, :]
            return x
        x = self.aggregation(x)
        return x


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
        if not args.unfreeze:
            for l in layers[:-5]:
                for p in l.parameters(): p.requires_grad = False
        if not args.unfreeze:
            logging.debug("Train last layers of the vgg16, freeze the previous ones")
        else:
            logging.debug("Train last layers of the vgg16")
    elif args.backbone == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        if not args.unfreeze:
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

def attach_projection_layers(args, backbone, aggregation, image_size, projection_size):
    rand_x = torch.randn(2, 3, image_size[0], image_size[1])
    backbone.eval()
    aggregation.eval()
    representation_before_agg = backbone(rand_x)
    representation_after_agg = aggregation(representation_before_agg)
    backbone.train()
    aggregation.train()
    _, dim = representation_after_agg.shape
    if args.ssl_method == "simsiam" or args.ssl_method == "byol":
        if args.n_layers == 3:
            fc_layer = nn.Sequential(nn.Linear(dim, projection_size),
                                    nn.BatchNorm1d(projection_size),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(projection_size, projection_size),
                                    nn.BatchNorm1d(projection_size),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(projection_size, projection_size),
                                    L2Norm())
        elif args.n_layers == 2:
            fc_layer = nn.Sequential(nn.Linear(dim, projection_size),
                                    nn.BatchNorm1d(projection_size),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(projection_size, projection_size),
                                    L2Norm())
        elif args.n_layers == 1:
            fc_layer = nn.Sequential(nn.Linear(dim, projection_size),
                                    L2Norm())
        else:
            raise NotImplementedError()
    elif args.ssl_method == "mocov2" or args.ssl_method == "simclr":
        if args.n_layers == 2:
            fc_layer = nn.Sequential(nn.Linear(dim, projection_size),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(projection_size, projection_size),
                                    L2Norm())
        elif args.n_layers == 1:
            fc_layer = nn.Sequential(nn.Linear(dim, projection_size),
                                    L2Norm())
        else:
            raise NotImplementedError()
    elif args.ssl_method == "vicreg" or args.ssl_method == "bt":
        if args.n_layers == 3:
            fc_layer = nn.Sequential(nn.Linear(dim, projection_size),
                                    nn.BatchNorm1d(projection_size),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(projection_size, projection_size),
                                    nn.BatchNorm1d(projection_size),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(projection_size, projection_size),
                                    L2Norm() if not args.remove_norm else nn.Identity()) 
        elif args.n_layers == 2:
            fc_layer = nn.Sequential(nn.Linear(dim, projection_size),
                                    nn.BatchNorm1d(projection_size),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(projection_size, projection_size),
                                    L2Norm() if not args.remove_norm else nn.Identity())
        elif args.n_layers == 1:
            fc_layer = nn.Sequential(nn.Linear(dim, projection_size),
                                    L2Norm() if not args.remove_norm else nn.Identity())
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    aggregation = nn.Sequential(
        aggregation,
        fc_layer
    )
    return aggregation

def visualize(image_tensor, name):
    if not os.path.isdir("vis"):
        os.mkdir("vis")
    for i in range(len(image_tensor)):
        img = image_tensor[i]
        img = inv_base_transforms(img)
        img.save(f"vis/{name}_{i}.png")

def setup_optimizer_loss(args, model_parameters, return_loss=True):
    # Setup Optimizer and Loss
    if args.aggregation == "crn":
        raise NotImplementedError()
    else:
        if args.optim == "adam":
            optimizer = torch.optim.Adam(model_parameters, lr=args.lr)
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD(
                model_parameters, lr=args.lr, momentum=0.9, weight_decay=0.001
            )
        else:
            raise NotImplementedError()
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
        self.backbone = get_backbone(args)
        self.args = args
        self.aggregation = get_aggregation(args)
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
        if args.n_layers > 0:
             self.aggregation = attach_projection_layers(args, self.backbone, self.aggregation, args.resize, args.projection_size)
        self.arch_name = args.backbone
        self.return_loss = False
        self.ssl_model = self.get_ssl_model()
        self.train_ds, self.val_ds, self.test_ds = ds_list
        self.all_features = None
        self.lr = 0

    def get_ssl_model(self):
        if self.args.ssl_method == "byol":
            self.return_loss = True
            return BYOL(self.backbone,
                        image_size = self.args.resize,
                        gpus_num = self.args.num_nodes * self.args.num_devices,
                        projection_size = self.args.projection_size if self.args.n_layers > 0 else self.args.features_dim,
                        aggregation = self.aggregation,
                        moving_average_decay = self.args.momentum,
                        n_layers = self.args.n_layers)
        elif self.args.ssl_method == "simsiam":
            self.return_loss = True
            return BYOL(self.backbone,
                        image_size = self.args.resize,
                        gpus_num = self.args.num_nodes * self.args.num_devices,
                        projection_size = self.args.projection_size if self.args.n_layers > 0 else self.args.features_dim,
                        aggregation = self.aggregation,
                        use_momentum=False,
                        n_layers = self.args.n_layers)
        elif self.args.ssl_method == "vicreg":
            self.return_loss = True
            return VICREG(self.backbone,
                        image_size = self.args.resize,
                        gpus_num = self.args.num_nodes * self.args.num_devices,
                        projection_size = self.args.projection_size if self.args.n_layers > 0 else self.args.features_dim,
                        aggregation = self.aggregation,
                        n_layers = self.args.n_layers)
        elif self.args.ssl_method == "bt":
            self.return_loss = True
            return VICREG(self.backbone,
                        image_size = self.args.resize,
                        gpus_num = self.args.num_nodes * self.args.num_devices,
                        projection_size = self.args.projection_size if self.args.n_layers > 0 else self.args.features_dim,
                        aggregation = self.aggregation,
                        use_bt_loss = True,
                        n_layers = self.args.n_layers)
        elif self.args.ssl_method == "mocov2":
            self.return_loss = True
            return MOCO(self.backbone,
                        image_size = self.args.resize,
                        gpus_num = self.args.num_nodes * self.args.num_devices,
                        projection_size = self.args.projection_size if self.args.n_layers > 0 else self.args.features_dim,
                        aggregation = self.aggregation,
                        K = self.args.queue_size,
                        n_layers = self.args.n_layers)
        elif self.args.ssl_method == "simclr":
            self.return_loss = True
            return MOCO(self.backbone,
                        image_size = self.args.resize,
                        gpus_num = self.args.num_nodes * self.args.num_devices,
                        projection_size = self.args.projection_size if self.args.n_layers > 0 else self.args.features_dim,
                        aggregation = self.aggregation,
                        use_simclr = True,
                        n_layers = self.args.n_layers)
        else:
            raise NotImplementedError()

    def forward(self, x, pairs_local_indexes=None, return_embedding=False, return_projection=True):
        if return_embedding:
            feature = self.ssl_model(x, None, return_embedding=True, return_projection=return_projection)
            return feature
        if pairs_local_indexes is None:
            raise NotImplementedError("pairs indexes should be pass if not return_feature")
        if self.args.pair_negative:
            x_indexes = pairs_local_indexes[0:len(pairs_local_indexes):3].long()
            y_indexes = pairs_local_indexes[1:len(pairs_local_indexes):3].long()
            input_x = x[x_indexes]
            input_y = x[y_indexes]
            for i in range(2, 3):
                neg_indexes = pairs_local_indexes[i:len(pairs_local_indexes):3].long()
                input_x = torch.cat([input_x, x[neg_indexes]], dim=0)
                input_y = torch.cat([input_y, x[neg_indexes]], dim=0)
        else:
            x_indexes = pairs_local_indexes[0:len(pairs_local_indexes):2].long()
            y_indexes = pairs_local_indexes[1:len(pairs_local_indexes):2].long()
            input_x = x[x_indexes]
            input_y = x[y_indexes]
        if self.args.visualize_input:
            visualize(input_x, "query")
            visualize(input_y, "positive")
            raise KeyError("Only visualize first batch. Comment this if you want more.")
        if self.return_loss:
            loss = self.ssl_model(input_x, input_y).mean()
            return loss
        else:
            raise NotImplementedError()
        
    def update(self):
        if self.args.ssl_method == "byol" or self.args.ssl_method == "mocov2":
            self.ssl_model.update_moving_average()
        elif self.args.ssl_method == "simsiam" or self.args.ssl_method == "vicreg" or self.args.ssl_method == "bt" or self.args.ssl_method == "simclr":
            pass
        else:
            raise NotImplementedError()
    
    def train_dataloader(self):
        # Compute pairs to use in the pair loss
        # SSL methods like vicreg and simclr requires drop last, otherwise the extra replicates will affect the loss
        self.train_ds.is_inference = True
        self.train_ds.compute_pairs(self.args, self.ssl_model)
        self.train_ds.is_inference = False
        pairs_dl = DataLoader(
            dataset=self.train_ds,
            num_workers=self.args.num_workers,
            batch_size=self.args.train_batch_size,
            collate_fn=datasets_ws.collate_fn_pair,
            pin_memory=False,
            drop_last=True,
            shuffle=True
        )

        return pairs_dl

    def val_dataloader(self):
        val_dataloader = DataLoader(
            dataset=self.val_ds,
            num_workers=self.args.num_workers,
            batch_size=self.args.infer_batch_size,
            pin_memory=False,
            shuffle=False
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            dataset=self.test_ds,
            num_workers=self.args.num_workers,
            batch_size=self.args.infer_batch_size,
            pin_memory=False,
            shuffle=False
        )
        return test_dataloader

    def training_step(self, inputs, _):
        if self.args.method == "pair":
            images, pairs_local_indexes, _, _ = inputs
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

            self.log("loss", loss_pairs, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.train_batch_size, sync_dist=True)
            self.log("current_lr", self.lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.train_batch_size, sync_dist=True)
            return {"loss": loss_pairs}
        else:
            raise NotImplementedError()
    
    def _shared_on_eval_epoch_start(self, eval_ds, args):
        if self.trainer.is_global_zero:
            if not self.args.eval_with_proj:
                self.all_features = torch.empty((len(eval_ds), self.args.features_dim), dtype=torch.float32, device="cpu")
            else:
                self.all_features = torch.empty((len(eval_ds), self.args.projection_size), dtype=torch.float32, device="cpu")

    def _shared_eval_step(self, eval_ds, inputs):
        images, indices = inputs
        features = self.forward(images, return_embedding=True, return_projection=self.args.eval_with_proj)
        features = self.all_gather(features)
        indices = self.all_gather(indices)
        if self.trainer.is_global_zero:
            features = features.view(-1, features.size(-1))
            indices = indices.view(-1)
            self.all_features[indices.cpu(), :] = features.cpu()

    def _shared_on_eval_epoch_end(self, eval_ds, args):
        queries_features = self.all_features[eval_ds.database_num:]
        database_features = self.all_features[: eval_ds.database_num]
        queries_features = F.normalize(queries_features, dim=1)
        database_features = F.normalize(database_features, dim=1)

        if args.eval_with_proj:
            faiss_index = faiss.IndexFlatL2(args.projection_size)
        else:
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
            logging.debug(f"val_recall5:{recalls[1]}")
            logging.debug(f"val_recall1:{recalls[0]}")
        else:
            recalls = np.zeros(len(self.args.recall_values))
        self.trainer.strategy.barrier()
        if self.trainer.num_devices == 1 and self.trainer.num_nodes == 1:
            recalls = recalls
        else:
            recalls = self.all_gather(recalls)
            recalls = recalls[0]
        self.log("val_recall5", recalls[1], on_epoch=True, logger=True)
        self.log("val_recall1", recalls[0], on_epoch=True, logger=True)
        

    def on_test_epoch_start(self):
        self._shared_on_eval_epoch_start(self.test_ds, self.args)

    def test_step(self, inputs, _):
        self._shared_eval_step(self.test_ds, inputs)

    def on_test_epoch_end(self):
        if self.trainer.is_global_zero:
            recalls, recalls_str = self._shared_on_eval_epoch_end(self.test_ds, self.args)
            logging.debug(f"test_recall5:{recalls[1]}")
            logging.debug(f"test_recall1:{recalls[0]}")
        else:
            recalls = np.zeros(len(self.args.recall_values))
        self.trainer.strategy.barrier()
        if self.trainer.num_devices == 1 and self.trainer.num_nodes == 1:
            recalls = recalls
        else:
            recalls = self.all_gather(recalls)
            recalls = recalls[0]
        self.log("test_recall5", recalls[1], logger=True)
        self.log("test_recall1", recalls[0], logger=True)

    def configure_optimizers(self):
        optimizer, self.criterion_pairs = setup_optimizer_loss(self.args, filter(lambda p: p.requires_grad, self.ssl_model.parameters()), return_loss=True)
        return optimizer

    def on_before_zero_grad(self, _):
        if self.args.cosine_scheduler:
            self.lr = adjust_learning_rate(self.optimizers(), self.current_epoch, self.args)
        self.update()

    def on_train_epoch_start(self):
        self.train_ds.is_inference = True
        self.train_ds.compute_pairs(self.args, self.ssl_model)
        self.train_ds.is_inference = False

    def setup(self, stage):
        if stage == "validate":
            self.initialize_flag = True
            if self.args.aggregation in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
                self.ssl_model.to(self.args.device)
                self.ssl_model.device = self.args.device
                if not self.args.resume:
                    self.train_ds.is_inference = True
                    if self.args.n_layers != 0:
                        self.aggregation[0].initialize_netvlad_layer(
                            self.args, self.train_ds, self.backbone)
                        self.args.features_dim = self.args.projection_size
                    else:
                        self.aggregation.initialize_netvlad_layer(
                            self.args, self.train_ds, self.backbone)
                        self.args.features_dim *= self.args.netvlad_clusters
                    self.train_ds.is_inference = False
        elif stage == "fit":
            if not hasattr(self, "initialize_flag"):
                self.initialize_flag = True
                if self.args.aggregation in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
                    self.ssl_model.to(self.args.device)
                    self.ssl_model.device = self.args.device
                    if not self.args.resume:
                        self.train_ds.is_inference = True
                        if self.args.n_layers != 0:
                            self.aggregation[0].initialize_netvlad_layer(
                                self.args, self.train_ds, self.backbone)
                            self.args.features_dim = self.args.projection_size
                        else:
                            self.aggregation.initialize_netvlad_layer(
                                self.args, self.train_ds, self.backbone)
                            self.args.features_dim *= self.args.netvlad_clusters
                        self.train_ds.is_inference = False
            self.ssl_model.to(self.args.device)
            self.ssl_model.device = self.args.device

    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure,
        ):
        optimizer_closure()
        optimizer.step()

class GeoLocalizationNetRerank(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.arch_name = args.backbone
        self.out_dim = args.local_dim
        if args.backbone.startswith("deit"):
            if args.backbone == 'deitBase':
                self.backbone = deit_base_distilled_patch16_384(img_size=args.resize, num_classes=args.fc_output_dim, embed_layer=AnySizePatchEmbed)
            else:
                self.backbone = deit_small_distilled_patch16_224(img_size=args.resize, num_classes=args.fc_output_dim, embed_layer=AnySizePatchEmbed)
            args.features_dim = args.fc_output_dim
            self.local_head = nn.Linear(self.backbone.embed_dim, self.out_dim, bias=True)
        else:
            self.backbone = get_backbone(args)
            self.aggregation = get_aggregation(args)
            self.self_att = False

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
                                                channel_inner=args.channel_bottleneck)] * args.num_non_local
                self.non_local = nn.Sequential(*non_local_list)
                self.self_att = True
            self.local_head = nn.Linear(1024, self.out_dim, bias=True)
        # ==================================================================
        self.local_head.weight.data.normal_(mean=0.0, std=0.01)
        self.local_head.bias.data.zero_()
        self.multi_out = args.num_local
        self.single = False
        if args.rerank_model == 'r2former':
            self.Reranker = R2Former(decoder_depth=6, decoder_num_heads=4,
                                     decoder_embed_dim=32, decoder_mlp_ratio=4,
                                     decoder_norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                     num_classes=2, num_patches=2 * self.multi_out,
                                     input_dim=args.fc_output_dim, num_corr=5)
        else:
            print('rerank_model not implemented!')
            raise Exception

    def forward_ori(self, x):
        x = self.backbone(x)
        if self.self_att:
            x = self.non_local(x)
        if self.arch_name.startswith("vit"):
            x = x.last_hidden_state[:, 0, :]
            return x
        x = self.aggregation(x)
        return x

    def res_forward(self, x):
        # print(self.backbone)
        # raise Exception
        x = self.backbone[0](x)
        x = self.backbone[1](x)
        x = self.backbone[2](x)
        x = self.backbone[3](x)
        x0 = x*1 #.detach()

        x = self.backbone[4](x)
        x1 = x*1 #.detach()
        x = self.backbone[5](x)
        x2 = x*1 #.detach()
        x = self.backbone[6](x)
        x3 = x*1
        x = self.backbone[7](x)
        x4 = x*1  #.detach()

        local_feature = x3

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x, local_feature, x4

    def forward_cnn(self, x):
        # with torch.no_grad():
        B,_, H, W = x.shape
        query_img = x.clone()
        x, feature, feature_last = self.res_forward(x)
        x = self.aggregation(x)

        _, C, f_H, f_W = feature.shape
        assert f_H == np.ceil(H/16).astype(int) and f_W == np.ceil(W/16).astype(int)
        feature_reshape = feature.permute((0,2,3,1)).reshape(B, f_H*f_W, C)
        # print(feature_last.shape, feature_reshape.shape, query_img.shape, H//32, W//32)
        feature_last_reshape = feature_last.permute((0,2,3,1)).reshape(B, np.ceil(H/32).astype(int)*np.ceil(W/32).astype(int), 2048)
        feature_last_reshape = F.normalize(feature_last_reshape, p=2, dim=2)
        # print(self.aggregation)
        fc_weight = self.aggregation[1].weight.t()
        # fc_weight = torch.eye(2048, dtype=torch.float32).cuda()
        sim = torch.matmul(feature_last_reshape.clamp(min=1e-6), fc_weight)
        last_map = (sim.clamp(min=1e-6)).sum(dim=2)  # /sim.max(dim=1,keepdim=True)[0]
        last_map_reshape = F.interpolate(last_map.reshape([B, 1, np.ceil(H/32).astype(int), np.ceil(W/32).astype(int)]),
                                        size=(np.ceil(H/16).astype(int), np.ceil(W/16).astype(int)), mode='bicubic')
        last_map = last_map_reshape.reshape(B, np.ceil(H/16).astype(int)*np.ceil(W/16).astype(int))
        # print(query_img.shape, x.shape, feature.shape, feature_reshape.shape)
        # print(sim.shape, last_map.shape, last_map_reshape.shape)

        order = torch.argsort(last_map, dim=1)
        multi_out = np.minimum(order.shape[1], self.multi_out)
        if order.shape[1] < self.multi_out:
            print(order.shape, last_map.shape, last_map)
        local_features = torch.gather(input=feature_reshape,
                                      index=order[:, -multi_out:].unsqueeze(2).repeat(1, 1, feature_reshape.shape[2]),
                                      dim=1)

        HW = max(H, W)
        # HW = 512.
        x_xy = torch.cat([(order[:, -multi_out:].unsqueeze(2) % np.ceil(W/16).astype(int) * 16 + 8) / 1. / HW,
                          (order[:, -multi_out:].unsqueeze(2) // np.ceil(W/16).astype(int) * 16 + 8) / 1. / HW], dim=2)
        x_attention = torch.sort(last_map, dim=1)[0][:, -multi_out:]
        x_attention = (x_attention / torch.max(x_attention, dim=1, keepdim=True)[0]).reshape(x_xy.shape[0],
                                                                                                 x_xy.shape[1], 1)
        if self.args.finetune:
            local_features = self.local_head(local_features.reshape(B*multi_out, C)).reshape(B, multi_out, self.out_dim)
        else:
            local_features = self.local_head(local_features.detach().reshape(B * multi_out, C)).reshape(B, multi_out,
                                                                                               self.out_dim)
        if self.single:
            return x
        else:
            return x, torch.flip(torch.cat([x_xy, x_attention, local_features], dim=2),dims=(1,))

    def forward_deit(self, x):
        # with torch.no_grad():
        B, _, H, W = x.shape
        x_ori = x.detach()
        x = self.backbone.patch_embed(x)

        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.backbone.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        if H != self.backbone.patch_embed.img_size[0] or W != self.backbone.patch_embed.img_size[1]:
            grid_size = [self.backbone.patch_embed.img_size[0]//16, self.backbone.patch_embed.img_size[1]//16]
            matrix = self.backbone.pos_embed[:, 2:].reshape((1, grid_size[0], grid_size[1],self.backbone.embed_dim)).permute((0, 3, 1, 2))
            new_size = max(H//16, W//16)
            if grid_size[0] >= new_size and grid_size[1] >= new_size:
                re_matrix = matrix[:, :, (grid_size[0]//2 - new_size//2):(grid_size[0]//2 - new_size//2 + new_size),
                            (grid_size[1]//2 - new_size//2):(grid_size[1]//2 - new_size//2+new_size)]
            else:
                re_matrix = pos_resize(matrix, (new_size, new_size))
            if H >= W:
                new_matrix = re_matrix[:, :, :, (new_size//2 - W//16//2):(new_size//2 - W//16//2 + W//16)].permute(0, 2, 3, 1).reshape([1, -1, self.backbone.pos_embed.shape[-1]])
            else:
                new_matrix = re_matrix[:, :, (new_size//2 - H//16//2):(new_size//2 - H//16//2 + H//16), :].permute(0, 2, 3, 1).reshape([1, -1, self.backbone.pos_embed.shape[-1]])
            # print(new_matrix.shape,H//16, W//16,new_size)
            new_pos_embed = torch.cat([self.backbone.pos_embed[:, :2], new_matrix], dim=1)
            x = x + new_pos_embed
        else:
            x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        output_list = []

        for i, blk in enumerate(self.backbone.blocks):
            if (not self.single) and i == (len(self.backbone.blocks)-1):  # len(self.blocks)-1:
                output = x * 1
                y = blk.norm1(x)
                B, N, C = y.shape
                qkv = blk.attn.qkv(y).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

                att = (q @ k.transpose(-2, -1)) * blk.attn.scale
                att = att.softmax(dim=-1)
                last_map = (att[:, :, :2, 2:].detach()).sum(dim=1).sum(dim=1)
            x = blk(x)

        x = self.backbone.norm(x)

        x_cls = self.backbone.head(x[:, 0])
        x_dist = self.backbone.head_dist(x[:, 1])

        if self.single:
            return self.backbone.l2_norm((x_cls + x_dist) / 2)
        else:
            order = torch.argsort(last_map, dim=1, descending=True)
            multi_out = np.minimum(order.shape[1], self.multi_out)
            local_features = torch.gather(input=output,
                                          index=order[:, :multi_out].unsqueeze(2).repeat(1, 1, output.shape[2]),
                                          dim=1)
            # compute attention and coordinates
            HW = max(H, W)
            x_xy = torch.cat([(order[:, :multi_out].unsqueeze(2) % np.ceil(W / 16).astype(int) * 16 + 8) / 1. / HW,
                              (order[:, :multi_out].unsqueeze(2) // np.ceil(W / 16).astype(int) * 16 + 8) / 1. / HW],
                             dim=2)
            x_attention = torch.sort(last_map, dim=1, descending=True)[0][:, :multi_out]
            x_attention = (x_attention / torch.max(x_attention, dim=1, keepdim=True)[0]).reshape(x_xy.shape[0],
                                                                                                 x_xy.shape[1], 1)
            if self.args.finetune:
                local_features = self.local_head(local_features.reshape(B * multi_out, -1)). \
                    reshape(B, multi_out, self.out_dim)
            else:
                local_features = self.local_head(local_features.detach().reshape(B * multi_out, -1)).\
                reshape(B, multi_out, self.out_dim)
            return self.backbone.l2_norm((x_cls + x_dist) / 2), torch.cat([x_xy, x_attention, local_features], dim=2)

    def forward(self, x):
        if self.args.backbone.startswith("deit"):
            return self.forward_deit(x)
        else:
            return self.forward_cnn(x)