import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_ch, 100),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=100),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=10),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.layer(x)
        print(x / 20)
        return torch.sigmoid(x)
