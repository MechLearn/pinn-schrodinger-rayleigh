# src/pinn1d/Harmonic/model.py
import torch
import torch.nn as nn

class HarmonicRayleighNet(nn.Module):
    """
    Misma convención que v2 paper 2:
    n empieza en 0
    n par   → función par  (psi = gauss * N(x))
    n impar → función impar (psi = gauss * x * N(x))
    Ecuación: -1/2·ψ'' + 1/2·x²·ψ = E·ψ
    E_exact = n + 0.5
    """
    def __init__(self, n=0, hidden=64, use_sine=True, L=6.0):
        super().__init__()
        self.n = n
        self.use_sine = use_sine
        self.L = L

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

        gauss = torch.exp(-0.5 * x ** 2)

        if self.n % 2 == 0:
            psi = gauss * out
        else:
            psi = gauss * x * out

        return psi
