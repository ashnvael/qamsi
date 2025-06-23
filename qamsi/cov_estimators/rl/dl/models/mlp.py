import torch
from torch import nn


class _MLP(nn.Module):
    def __init__(self, sizes: list[int]):
        super().__init__()
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                layers.append(nn.LeakyReLU(0.1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLPPredictor(nn.Module):
    def __init__(
        self, hidden_size: int, n_features: int, n_layers: int, *args, **kwargs
    ):
        super().__init__()

        self.model = _MLP([n_features] + ([hidden_size] * n_layers) + [1])

    def forward(self, x: torch.Tensor):
        return self.model(x)
