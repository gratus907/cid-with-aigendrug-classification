import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_ch, 881),
            nn.Tanh(),
            nn.Linear(881, 440),
            nn.Tanh(),
            nn.Linear(440, 110),
            nn.Tanh(),
            nn.Linear(110, 1)
        )

    def forward(self, x):
        x = self.layer(x)
        return torch.sigmoid(x)
