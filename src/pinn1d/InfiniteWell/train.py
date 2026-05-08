# src/pinn1d/InfiniteWell/train.py
import os
import math
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from .model import RayleighNet
from .losses import compute_losses


def run_one_mode(n: int, cfg: dict):
    base_dir = cfg.get("save_dir", "outputs/runs_rayleigh")
    exp_name = cfg.get("experiment_name", "rayleigh")
    seed     = cfg.get("seed", 0)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(base_dir, f"{exp_name}_seed{seed}", f"mode_{n}")
    os.makedirs(save_dir, exist_ok=True)

    HIDDEN   = cfg["model"]["hidden"]
    USE_SINE = cfg["model"]["use_sine"]
    N_col    = max(1024, 2048 * n)
    EPOCHS   = 15000 if n >= 4 else (9000 if n == 3 else (6000 if n == 2 else 4000))
    LR0      = 3e-4  if n >= 4 else (5e-4 if n == 3 else (7e-4 if n == 2 else 1e-3))
    lam_hi, lam_lo = (300.0, 80.0) if n >= 3 else (40.0, 15.0 if n == 2 else 10.0)

    net = RayleighNet(n=n, hidden=HIDDEN, use_sine=USE_SINE).to(device)

    x_col   = np.linspace(0, 1, N_col, dtype=np.float32).reshape(-1, 1)
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
        L, LPDE, Lnorm, integral, E = compute_losses(net, x_batch, lam)
        L.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if ep <= 500 or ep % 200 == 0 or ep == EPOCHS:
            loss_total.append(float(L))
            loss_pde.append(float(LPDE))
            loss_norm.append(float(Lnorm))
            E_hist.append(float(E))
            int_hist.append(float(integral))
            epochs_logged.append(ep)

        if ep % max(1000, EPOCHS // 5) == 0 or ep == 1:
            print(
                f"n={n} ep={ep} "
                f"E={float(E):.6f} "
                f"L={float(L):.3e} "
                f"LPDE={float(LPDE):.3e} "
                f"Lnorm={float(Lnorm):.3e} "
                f"integral={float(integral):.6f} "
                f"lam={lam:.1f}"
            )

    # Evaluación final
    net.eval()
    with torch.no_grad():
        xs       = torch.linspace(0, 1, 2000, device=device).reshape(-1, 1)
        psi_pred = net(xs).cpu().numpy().squeeze()

    xs_np     = np.linspace(0, 1, 2000)
    psi_exact = np.sqrt(2.0) * np.sin(n * math.pi * xs_np)

    sign      = np.sign(np.dot(psi_pred, psi_exact))
    psi_pred *= sign

    l2_err    = float(np.sqrt(np.mean((psi_pred - psi_exact) ** 2)))
    integ     = float(np.trapezoid(psi_pred ** 2, xs_np))
    E_learned = float(E)
    E_exact   = float((n * math.pi) ** 2)
    E_rel     = float(abs(E_learned - E_exact) / (abs(E_exact) + 1e-12))

    # Figura función de onda
    plt.figure(figsize=(7, 4))
    plt.plot(xs_np, psi_pred,  label="PINN")
    plt.plot(xs_np, psi_exact, "--", label="Exacta")
    plt.title(f"n={n} | E={E_learned:.6f} | rel={E_rel:.2e} | L2={l2_err:.2e} | integral={integ:.4f}")
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
        "potential":       "infinite_well_rayleigh",
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
        loss_norm     = np.array(loss_norm,      dtype=np.float64),
        E_hist        = np.array(E_hist,        dtype=np.float64),
        int_hist      = np.array(int_hist,      dtype=np.float64),
        epochs_logged = np.array(epochs_logged, dtype=np.int32),
    )

    return {
        "n":            n,
        "save_dir":     save_dir,
        "E_learned":    E_learned,
        "E_exact":      E_exact,
        "E_rel":        E_rel,
        "L2":           l2_err,
        "integral":     integ,
        "device":       str(device),
    }
