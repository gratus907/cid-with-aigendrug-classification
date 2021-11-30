import torch
import torch.nn as nn

class SVM(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.linear = nn.Linear(in_ch, 1)

    def forward(self, x):
        x = self.linear(x)
        return torch.sigmoid(x)
