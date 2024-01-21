import torch
import torch.nn as nn

class Sin(torch.nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class ReLU_Feat_Posenc(nn.Module):
    def __init__(self, N, M, frequencies=16):
        super(ReLU_Feat_Posenc, self).__init__()

        self.FEATURE_SIZE = 100
        self.FEATURE_GRID = (N, M)
        self.features = nn.Parameter(torch.rand(*self.FEATURE_GRID, self.FEATURE_SIZE))
        self.frequencies = frequencies

        ffin = self.FEATURE_SIZE + 2 * (2 * self.frequencies + 1)
        self.model = nn.Sequential(
            nn.Linear(ffin, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 3)
        )

    def posenc(self, UV):
        X = [ UV ]
        for i in range(self.frequencies):
            k = 2 ** i
            X += [ torch.sin(k * UV), torch.cos(k * UV) ]

        return torch.cat(X, dim=-1)

    def forward(self, UV):
        U, V = UV[..., 0], UV[..., 1]
        iU = ((self.FEATURE_GRID[0] - 1) * U)
        iV = ((self.FEATURE_GRID[1] - 1) * V)
        iU0, iU1 = iU.floor().int(), iU.ceil().int()
        iV0, iV1 = iV.floor().int(), iV.ceil().int()

        iU = (iU - iU0).unsqueeze(-1)
        iV = (iV - iV0).unsqueeze(-1)

        f00 = self.features[iU0, iV0] * (1.0 - iU) * (1.0 - iV)
        f01 = self.features[iU1, iV0] * iU * (1.0 - iV)
        f10 = self.features[iU0, iV1] * (1.0 - iU) * iV
        f11 = self.features[iU1, iV1] * iU * iV
        features = f00 + f01 + f10 + f11

        return self.model(torch.cat((self.posenc(UV), features), dim=-1))

