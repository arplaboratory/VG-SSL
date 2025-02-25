import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import transforms as T
from model.byol.byol_pytorch import get_module_device
import logging

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


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def create_projector(embedding_dim, mlp):
    mlp_spec = f"{embedding_dim}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


class VICREG(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        gpus_num,
        projection_size = 8192,
        projection_hidden_size = 8192,
        use_bt_loss = False,
        sim_coeff = 25.0,
        std_coeff = 25.0,
        cov_coeff = 1.0,
        lambd = 0.0051,
        n_layers = 3,
        aggregation = None,
    ):
        super().__init__()
        self.net = net
        self.aggregation = aggregation
        self.gpus_num = gpus_num
        # Augmentation is finished outside

        self.use_bt_loss = use_bt_loss
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.lambd = lambd
        self.mlp = ""
        for i in range(n_layers-1):
            self.mlp += str(projection_hidden_size) + "-"
        self.mlp += str(projection_size)
        self.num_features = int(self.mlp.split("-")[-1])
        
        if n_layers > 0:
            self.aggregation_before_proj = self.aggregation[0]
        else:
            self.aggregation_before_proj = self.aggregation
        self.bn = None

        # get device of network and make wrapper same device
        self.device = get_module_device(self.net)
        self.to(self.device)

        # send a mock image tensor to instantiate singleton parameters
        # no need for bt and vicreg
        # self.eval()
        # self.forward(torch.randn(2, 3, image_size[0], image_size[1], device=self.device), torch.randn(2, 3, image_size[0], image_size[1], device=self.device))
        # self.train()

    def forward(
        self,
        x,
        y,
        return_embedding = False,
        return_projection = True
    ):
        if return_embedding:
            if not return_projection:
                return self.aggregation_before_proj(self.net(x))
            else:
                return self.aggregation(self.net(x))

        x = self.net(x)
        y = self.net(y)

        x = self.aggregation(x)
        y = self.aggregation(y)

        if not self.use_bt_loss:
            # Use vicreg loss
            repr_loss = F.mse_loss(x, y)

            if self.gpus_num > 1:
                x = torch.cat(FullGatherLayer.apply(x), dim=0)
                y = torch.cat(FullGatherLayer.apply(y), dim=0)
            x = x - x.mean(dim=0)
            y = y - y.mean(dim=0)

            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

            batch_size = x.shape[0] # Since gathered, batch size is global
            cov_x = (x.T @ x) / (batch_size - 1)
            cov_y = (y.T @ y) / (batch_size - 1)
            cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
                self.num_features
            ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

            loss = (
                self.sim_coeff * repr_loss
                + self.std_coeff * std_loss
                + self.cov_coeff * cov_loss
            )
        else:
            # Use Barlow twins loss
            # empirical cross-correlation matrix
            if self.gpus_num > 1:
                x = torch.cat(FullGatherLayer.apply(x), dim=0)
                y = torch.cat(FullGatherLayer.apply(y), dim=0)
            x = x - x.mean(dim=0)
            y = y - y.mean(dim=0)
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            x = x / std_x
            y = y / std_y
            c = x.T @ y

            # sum the cross-correlation matrix between all gpus
            batch_size = x.shape[0] # Since gathered, batch size is global
            c.div_(batch_size)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss = on_diag + self.lambd * off_diag

        return loss