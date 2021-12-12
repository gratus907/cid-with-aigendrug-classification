import torch
import torch.nn as nn

class Weight(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(data = torch.Tensor(2621), requires_grad=True)
        self.weight.data.uniform_(1, 1)

    def forward(self, x):
        return torch.mul(x, self.weight)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = Weight()
        self.layer = nn.Sequential(
            nn.Linear(2621, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.weight(x)
        x = self.layer(x)
        return torch.sigmoid(x)
