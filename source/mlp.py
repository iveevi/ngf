import torch.nn as nn

from typing import Callable

class MLP(nn.Module):
    def __init__(self, ffin: int, activation: Callable) -> None:
        super(MLP, self).__init__()

        # Pass configuration as constructor arguments
        # TODO: reduce to a seq model of relu or custom sin
        self.encoding_linears = [
            nn.Linear(ffin, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
            # TODO: disable bias in the last layer?
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)
        self.activation = activation

    def forward(self, bases, X):
        X = self.encoding_linears[0](X)
        X = self.activation(X)

        X = self.encoding_linears[1](X)
        X = self.activation(X)

        X = self.encoding_linears[2](X)
        X = self.activation(X)

        X = self.encoding_linears[3](X)

        return bases + X
