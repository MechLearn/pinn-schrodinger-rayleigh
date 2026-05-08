# src/pinn1d/InfiniteWell/model.py
import math
import torch
import torch.nn as nn

def trig_nodal_factor(x, n):
    s1 = torch.sin(math.pi * x)
    sn = torch.sin(n * math.pi * x)
    ratio = sn / (s1 + 1e-12)
    return torch.where(torch.abs(s1) < 1e-6, torch.full_like(x, float(n)), ratio)

class RayleighNet(nn.Module):
    def __init__(self, n=1, hidden=64, use_sine=True):
        super().__init__()
        self.n = n
        self.use_sine = use_sine
        act = torch.sin if use_sine else torch.tanh

        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        if self.use_sine:
            z = torch.sin(self.fc1(x))
            z = torch.sin(self.fc2(z))
        else:
            z = torch.tanh(self.fc1(x))
            z = torch.tanh(self.fc2(z))
        out = self.fc3(z)
        F = trig_nodal_factor(x, self.n)
        psi = x * (1.0 - x) * F * out
        return psi
