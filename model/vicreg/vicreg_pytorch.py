import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import transforms as T
from model.byol.byol_pytorch import get_module_device

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
        batch_size,
        use_bt_loss = False,
        sim_coeff = 25.0,
        std_coeff = 25.0,
        cov_coeff = 1.0,
        lambd = 0.0051,
        mlp = "8192-8192-8192",
        aggregation = None,
        device = "cuda"
    ):
        super().__init__()
        self.net = net
        self.aggregation = aggregation
        self.use_bt_loss = use_bt_loss
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.lambd = lambd
        self.num_features = int(mlp.split("-")[-1])
        self.mlp = mlp
        # Augmentation is finished outside

        self.projector = None
        self.bn = None
        self.batch_size = batch_size

        # get device of network and make wrapper same device
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.eval()
        self.forward(torch.randn(2, 3, image_size[0], image_size[1], device=device), torch.randn(2, 3, image_size[0], image_size[1], device=device))
        self.train()

    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = create_projector(dim, self.mlp)
        return projector.to(hidden)

    def _get_bn(self, hidden):
        _, dim = hidden.shape
        bn = nn.BatchNorm1d(dim, affine=False)
        return bn.to(hidden)

    def forward(
        self,
        x,
        y,
        return_embedding = False,
        return_projection = True
    ):
        if return_embedding:
            return self.aggregation(self.net(x))

        x = self.net(x)
        y = self.net(y)

        x = self.aggregation(x)
        y = self.aggregation(y)

        if self.projector is None:
            self.projector = self._get_projector(x)
            if self.use_bt_loss:
                self.bn = self._get_bn(x)
            return

        x = self.projector(x)
        y = self.projector(y)

        if not self.use_bt_loss:
            # Use vicreg loss
            repr_loss = F.mse_loss(x, y)

            x = torch.cat(FullGatherLayer.apply(x), dim=0)
            y = torch.cat(FullGatherLayer.apply(y), dim=0)
            x = x - x.mean(dim=0)
            y = y - y.mean(dim=0)

            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

            cov_x = (x.T @ x) / (self.batch_size - 1)
            cov_y = (y.T @ y) / (self.batch_size - 1)
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
            c = self.bn(z1).T @ self.bn(z2)

            # sum the cross-correlation matrix between all gpus
            c.div_(self.batch_size)
            torch.distributed.all_reduce(c)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss = on_diag + self.lambd * off_diag

        return loss