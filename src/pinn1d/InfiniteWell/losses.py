# src/pinn1d/InfiniteWell/losses.py
import torch
from .derivatives import second_derivative

def compute_losses(model, x_batch, lam):
    """
    Pérdida para el pozo infinito con cociente de Rayleigh.
    
    Diferencia clave vs formulación estándar:
    - NO hay parámetro alpha libre
    - E se calcula directamente de psi via cociente de Rayleigh
    - E_R = -∫ψ·ψ'' dx / ∫ψ² dx  (V=0 para pozo infinito)
    """
    psi, psi_xx = second_derivative(model, x_batch)

    psi_sq    = psi.squeeze() ** 2
    psi_val   = psi.squeeze()
    psixx_val = psi_xx.squeeze()
    xb        = x_batch.squeeze()
    dx        = xb[1:] - xb[:-1]

    # Denominador: ∫ψ² dx  (trapecio)
    denominator = torch.sum(0.5 * (psi_sq[1:] + psi_sq[:-1]) * dx)

    # Numerador: -∫ψ·ψ'' dx  (trapecio)
    psi_psixx = psi_val * psixx_val
    numerator = -torch.sum(0.5 * (psi_psixx[1:] + psi_psixx[:-1]) * dx)

    # Cociente de Rayleigh
    E_rayleigh = numerator / (denominator + 1e-8)

    # Residuo PDE
    res  = psi_xx + E_rayleigh * psi
    LPDE = torch.mean(res ** 2)

    # Normalización
    integral = denominator
    Lnorm    = (integral - 1.0) ** 2

    L = LPDE + lam * Lnorm
    return L, LPDE, Lnorm, integral, E_rayleigh
