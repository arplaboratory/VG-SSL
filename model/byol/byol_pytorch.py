# https://raw.githubusercontent.com/lucidrains/byol-pytorch/master/byol_pytorch/byol_pytorch.py
import copy
import random
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
import logging

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor
def NoBnMLP(dim, projection_size, hidden_size=4096, n_layers=2):
    if n_layers == 1:
        return nn.Sequential(
            nn.Linear(dim, projection_size)
        )
    elif n_layers == 2:
        return nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )
    else:
        raise NotImplementedError()

def MLP(dim, projection_size, hidden_size=4096, n_layers=2):
    if n_layers == 1:
        return nn.Sequential(
            nn.Linear(dim, projection_size)
        )
    elif n_layers == 2:
        return nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )
    else:
        raise NotImplementedError()

def SimSiamMLP(dim, projection_size, hidden_size=4096, n_layers=3):
    if n_layers == 3:
        return nn.Sequential(
            nn.Linear(dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size, bias=False),
            nn.BatchNorm1d(projection_size, affine=False)
        )
    elif n_layers == 2:
        return nn.Sequential(
            nn.Linear(dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size, bias=False),
            nn.BatchNorm1d(projection_size, affine=False)
        )
    elif n_layers == 1:
        return nn.Sequential(
            nn.Linear(dim, projection_size, bias=False),
            nn.BatchNorm1d(projection_size, affine=False)
        )
    else:
        raise NotImplementedError()

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, mlp = "MLP", aggregation = None, n_layers = -1):
        super().__init__()
        self.net = net
        self.aggregation = aggregation
        self.n_layers = n_layers

        if self.n_layers > 0:
            self.aggregation_before_proj = self.aggregation[0]
        else:
            self.aggregation_before_proj = self.aggregation
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        if mlp in ["MLP", "SimsiamMLP", "NoBnMLP"]:
            self.mlp = mlp
        else:
            raise NotImplementedError()

    def get_representation(self, x, return_projection = True):
        if return_projection:
            return self.aggregation(self.net(x))
        else:
            return self.aggregation_before_proj(self.net(x))

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x, return_projection)
        return representation

# main class

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        gpus_num,
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99,
        use_momentum = True,
        aggregation = None,
        n_layers = -1
    ):
        super().__init__()
        self.net = net
        self.aggregation = aggregation
        self.gpus_num = gpus_num
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        # Augmentation is finished outside

        # Default layer num for projector
        if n_layers == -1:
            if use_momentum:
                n_layers = 2
            else:
                n_layers = 3

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, mlp="MLP" if use_momentum else "SimsiamMLP", aggregation=self.aggregation, n_layers = n_layers)
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = None

        # get device of network and make wrapper same device
        self.device = get_module_device(self.net)
        self.to(self.device)

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

        online_proj_one = self.online_encoder(image_one, return_projection = True)
        online_proj_two = self.online_encoder(image_two, return_projection = True)

        if self.online_predictor is None:
            _, dim = online_proj_one.shape
            self.online_predictor = MLP(dim, self.projection_size, self.projection_hidden_size)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            if self.target_encoder is None:
                self.target_encoder = self._get_target_encoder()
            
            target_encoder =  self.target_encoder if self.use_momentum else self.online_encoder
            target_proj_one = target_encoder(image_one, return_projection = True)
            target_proj_two = target_encoder(image_two, return_projection = True)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()