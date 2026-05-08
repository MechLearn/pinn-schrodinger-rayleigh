# src/pinn1d/InfiniteWell/derivatives.py
import torch

def second_derivative(model, x):
    """
    Calcula psi y psi_xx usando autograd de PyTorch.
    x debe tener requires_grad=True
    """
    x = x.requires_grad_(True)
    psi = model(x)
    
    psi_x = torch.autograd.grad(
        psi, x,
        grad_outputs=torch.ones_like(psi),
        create_graph=True
    )[0]
    
    psi_xx = torch.autograd.grad(
        psi_x, x,
        grad_outputs=torch.ones_like(psi_x),
        create_graph=True
    )[0]
    
    return psi, psi_xx
