import torch
import torch.nn as nn

class NeuralMulticharts(nn.Module):
    LEVELS = 16
    FEATURE_SIZE = 500

    def __init__(self):
        super(NeuralMulticharts, self).__init__()

        ffin = 2 * (2 * NeuralMulticharts.LEVELS + 1) + NeuralMulticharts.FEATURE_SIZE
        self.model = nn.Sequential(
                nn.Linear(ffin, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 3)
        )

    def encoding(self, X, F):
        Xs = [ F, X ]
        for i in range(NeuralMulticharts.LEVELS):
            k = 2 ** i
            Xs += [ torch.sin(k * X), torch.cos(k * X) ]

        return torch.cat(Xs, dim=-1)

    def forward(self, UV, F):
        return self.model(self.encoding(UV, F))

    def evaluate(self, **kwargs):
        features = kwargs['features']
        sampling = kwargs['sampling']

        with torch.no_grad():
            U = torch.linspace(0, 1, sampling[0] * sampling[2])
            V = torch.linspace(0, 1, sampling[1] * sampling[2])
            U, V = torch.meshgrid(U, V, indexing='ij')
            UV_whole = torch.stack([U, V], dim=-1).cuda()
            translated = features.repeat((sampling[2], sampling[2], 1))
            return self.forward(UV_whole, translated)
