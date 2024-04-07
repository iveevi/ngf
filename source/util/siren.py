import torch
import torch.nn as nn
import numpy as np


class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(self, ffin: int):
        super().__init__()
        self.layers = SIREN.generate_layers(ffin, 64, 2, 3)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

    def stream(self):
        weights = [s.linear.weight.data.cpu() for s in self.layers if isinstance(s, SirenLayer)]
        biases = [s.linear.bias.data.cpu() for s in self.layers if isinstance(s, SirenLayer)]

        bytestream = b''
        for w in weights:
            bytestream += w.shape[0].to_bytes(4, 'little')
            bytestream += w.shape[1].to_bytes(4, 'little')
            bytestream += w.numpy().astype('float32').tobytes()

        for b in biases:
            bytestream += b.shape[0].to_bytes(4, 'little')
            bytestream += b.numpy().astype('float32').tobytes()

        return bytestream

    @staticmethod
    def generate_layers(isize: int,
                        hidden: int,
                        layers: int,
                        osize: int,
                        outermost_linear=True,
                        first_omega_0=30,
                        hidden_omega_0=30) -> list[nn.Module]:
        net = [SirenLayer(isize, hidden, is_first=True, omega_0=first_omega_0)]
        for i in range(layers):
            net.append(SirenLayer(hidden, hidden, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden, osize)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden) / hidden_omega_0, np.sqrt(6 / hidden) / hidden_omega_0)
            net.append(final_linear)
        else:
            net.append(SirenLayer(hidden, osize, is_first=False, omega_0=hidden_omega_0))

        return net