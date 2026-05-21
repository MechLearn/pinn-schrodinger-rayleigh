# src/pinn1d/Harmonic/losses.py
import torch
from .derivatives import second_derivative

def compute_losses(model, x_batch, lam):
    """
    H = -1/2·d²/dx² + 1/2·x²
    E_R = (-1/2·∫ψ·ψ'' dx + 1/2·∫ψ²·x² dx) / ∫ψ² dx
    E_exact = n + 0.5
    
    Nota: Sin restricción de ortogonalidad la red puede converger
    a un modo diferente al pedido (modo incorrecto). Requiere
    deflación para garantizar el modo n específico.
    """
    psi, psi_xx = second_derivative(model, x_batch)

    psi_sq    = psi.squeeze() ** 2
    psi_val   = psi.squeeze()
    psixx_val = psi_xx.squeeze()
    xb        = x_batch.squeeze()
    dx        = xb[1:] - xb[:-1]

    denominator = torch.sum(0.5 * (psi_sq[1:] + psi_sq[:-1]) * dx)

    psi_psixx = psi_val * psixx_val
    kinetic   = -0.5 * torch.sum(0.5 * (psi_psixx[1:] + psi_psixx[:-1]) * dx)

    V         = 0.5 * xb ** 2
    psi_sq_V  = psi_sq * V
    potential = torch.sum(0.5 * (psi_sq_V[1:] + psi_sq_V[:-1]) * dx)

    E_rayleigh = (kinetic + potential) / (denominator + 1e-8)

    res  = -0.5 * psixx_val + V * psi_val - E_rayleigh * psi_val
    LPDE = torch.mean(res ** 2)

    integral = denominator
    Lnorm    = (integral - 1.0) ** 2

    L = LPDE + lam * Lnorm
    return L, LPDE, Lnorm, integral, E_rayleigh
