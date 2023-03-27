import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import transforms as T

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


def create_projector(embedding_dim):
    mlp_spec = f"{embedding}-{mlp}"
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
        batch_size = 1,
        aggregation = None,
        sim_coeff = 25.0,
        std_coeff = 25.0,
        cov_coeff = 1.0,
        num_features = 2048,
        mlp = "2048-2048",
    ):
        super().__init__()
        self.net = net
        self.aggregation = aggregation
        # Augmentation is finished outside

        self.projector = None
        self.batch_size = batch_size

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size[0], image_size[1], device=device), torch.randn(2, 3, image_size[0], image_size[1], device=device))

    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = create_projector(dim)
        return projector.to(hidden)

     def forward(
        self,
        x,
        y,
        return_embedding = False,
        return_projection = True
    ):
        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)

        x = self.backbone(x)
        y = self.backbone(y)

        x = self.aggregation(x)
        y = self.aggregation(y)

        if self.projector is None:
            self.projector = self._get_projector(x)
        x = self.projector(x)
        y = self.projector(y)

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
            num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(num_features)

        loss = (
            sim_coeff * repr_loss
            + std_coeff * std_loss
            + cov_coeff * cov_loss
        )
        return loss