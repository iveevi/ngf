import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, ffin: int) -> None:
        super(MLP, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(ffin, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, X):
        return self.seq(X)
