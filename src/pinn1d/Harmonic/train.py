# src/pinn1d/Harmonic/train.py
import os
import math
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.special import factorial

from .model import HarmonicRayleighNet
from .losses import compute_losses


def exact_harmonic(x, n):
    """
    Eigenfunción exacta del oscilador armónico 1D (unidades naturales ℏ=m=ω=1)
    ψn(x) = (2^n · n! · √π)^(-1/2) · Hn(x) · exp(-x²/2)
    En = n + 1/2
    """
    Hn = hermite(n)
    norm = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
    return norm * Hn(x) * np.exp(-x**2 / 2.0)


def run_one_mode(n: int, cfg: dict):
    base_dir = cfg.get("save_dir", "outputs/runs_rayleigh_harmonic")
    exp_name = cfg.get("experiment_name", "rayleigh_harmonic")
    seed     = cfg.get("seed", 0)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(base_dir, f"{exp_name}_seed{seed}", f"mode_{n}")
    os.makedirs(save_dir, exist_ok=True)

    HIDDEN   = cfg["model"]["hidden"]
    USE_SINE = cfg["model"]["use_sine"]
    L        = cfg.get("domain_L", 6.0)
    N_col    = max(1024, 2048 * (n + 1))
    EPOCHS   = 15000 if n >= 4 else (9000 if n == 3 else (6000 if n == 2 else 4000))
    LR0      = 3e-4  if n >= 4 else (5e-4 if n == 3 else (7e-4 if n == 2 else 1e-3))
    lam_hi, lam_lo = (300.0, 80.0) if n >= 3 else (40.0, 15.0 if n == 2 else 10.0)

    net = HarmonicRayleighNet(n=n, hidden=HIDDEN, use_sine=USE_SINE, L=L).to(device)

    x_col   = np.linspace(-L, L, N_col, dtype=np.float32).reshape(-1, 1)
    x_batch = torch.tensor(x_col, device=device)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR0)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=EPOCHS, power=1.0
    )

    lam = lam_hi

    loss_total, loss_pde, loss_norm = [], [], []
    E_hist, int_hist, epochs_logged = [], [], []

    for ep in range(1, EPOCHS + 1):
        if ep == EPOCHS // 3:
            lam = lam_lo

        optimizer.zero_grad()
        L_val, LPDE, Lnorm, integral, E = compute_losses(net, x_batch, lam)
        L_val.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if ep <= 500 or ep % 200 == 0 or ep == EPOCHS:
            loss_total.append(float(L_val))
            loss_pde.append(float(LPDE))
            loss_norm.append(float(Lnorm))
            E_hist.append(float(E))
            int_hist.append(float(integral))
            epochs_logged.append(ep)

        if ep % max(1000, EPOCHS // 5) == 0 or ep == 1:
            print(
                f"n={n} ep={ep} "
                f"E={float(E):.6f} "
                f"L={float(L_val):.3e} "
                f"LPDE={float(LPDE):.3e} "
                f"Lnorm={float(Lnorm):.3e} "
                f"integral={float(integral):.6f} "
                f"lam={lam:.1f}"
            )

    # Evaluación final
    net.eval()
    with torch.no_grad():
        xs_t     = torch.linspace(-L, L, 2000, device=device).reshape(-1, 1)
        psi_pred = net(xs_t).cpu().numpy().squeeze()

    xs_np     = np.linspace(-L, L, 2000)
    psi_exact = exact_harmonic(xs_np, n)

    # Alinear signo
    sign      = np.sign(np.dot(psi_pred, psi_exact))
    psi_pred *= sign

    l2_err    = float(np.sqrt(np.mean((psi_pred - psi_exact) ** 2)))
    integ     = float(np.trapezoid(psi_pred ** 2, xs_np))
    E_learned = float(E)
    E_exact   = float(n + 0.5)
    E_rel     = float(abs(E_learned - E_exact) / (abs(E_exact) + 1e-12))

    # Figura función de onda
    plt.figure(figsize=(7, 4))
    plt.plot(xs_np, psi_pred,  label="PINN")
    plt.plot(xs_np, psi_exact, "--", label="Exacta")
    plt.title(f"n={n} | E={E_learned:.6f} | E_exact={E_exact:.4f} | rel={E_rel:.2e} | L2={l2_err:.2e}")
    plt.xlabel("x"); plt.ylabel("psi"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mode.png"), dpi=150); plt.close()

    # Figura loss
    plt.figure(figsize=(7, 4))
    plt.semilogy(epochs_logged, loss_total, label="Total")
    plt.semilogy(epochs_logged, loss_pde,   label="PDE")
    plt.semilogy(epochs_logged, loss_norm,  label="Norm")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss.png"), dpi=150); plt.close()

    metrics = {
        "potential":       "harmonic_rayleigh",
        "n":               n,
        "experiment_name": exp_name,
        "seed":            int(seed),
        "E_learned":       E_learned,
        "E_exact":         E_exact,
        "E_rel":           E_rel,
        "L2":              l2_err,
        "integral":        integ,
        "final_loss":      loss_total[-1],
        "final_pde":       loss_pde[-1],
        "final_norm":      loss_norm[-1],
        "epochs":          EPOCHS,
        "N_col":           N_col,
        "hidden":          HIDDEN,
        "use_sine":        USE_SINE,
        "domain_L":        L,
        "lam_hi":          lam_hi,
        "lam_lo":          lam_lo,
        "lr0":             LR0,
        "device":          str(device),
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        os.path.join(save_dir, "history.npz"),
        loss_total    = np.array(loss_total,    dtype=np.float64),
        loss_pde      = np.array(loss_pde,      dtype=np.float64),
        loss_norm     = np.array(loss_norm,     dtype=np.float64),
        E_hist        = np.array(E_hist,        dtype=np.float64),
        int_hist      = np.array(int_hist,      dtype=np.float64),
        epochs_logged = np.array(epochs_logged, dtype=np.int32),
    )

    return {
        "n":          n,
        "save_dir":   save_dir,
        "E_learned":  E_learned,
        "E_exact":    E_exact,
        "E_rel":      E_rel,
        "L2":         l2_err,
        "integral":   integ,
        "device":     str(device),
    }
