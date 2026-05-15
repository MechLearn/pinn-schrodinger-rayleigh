# scripts/analyze_sweep.py
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml

def load_results(base_dir, exp_name, seeds, n_min, n_max):
    results = {}
    for n in range(n_min, n_max + 1):
        results[n] = []
        for seed in seeds:
            path = os.path.join(
                base_dir,
                f"{exp_name}_seed{seed}",
                f"mode_{n}",
                "metrics.json"
            )
            if os.path.exists(path):
                with open(path) as f:
                    results[n].append(json.load(f))
    return results

def collapse_rate_wilson(collapsed, total, alpha=0.05):
    if total == 0:
        return None, None, None
    rate = collapsed / total
    z = 1.96
    center = (collapsed + z**2/2) / (total + z**2)
    margin = z * np.sqrt((collapsed*(total-collapsed)/total + z**2/4)) / (total + z**2)
    return rate, max(0, center - margin), min(1, center + margin)

def analyze(cfg_a, cfg_b=None, seeds=None, n_min=1, n_max=30, threshold=0.1, out_dir="outputs/analysis"):
    os.makedirs(out_dir, exist_ok=True)
    seeds = seeds or list(range(5))

    res_a = load_results(cfg_a["save_dir"], cfg_a["experiment_name"], seeds, n_min, n_max)
    res_b = load_results(cfg_b["save_dir"], cfg_b["experiment_name"], seeds, n_min, n_max) if cfg_b else None

    modes = list(range(n_min, n_max + 1))

    rates_a, lo_a, hi_a = [], [], []
    rates_b, lo_b, hi_b = [], [], []
    E_rel_a, E_rel_b = [], []

    for n in modes:
        total_a = len(res_a[n])
        collapsed_a = sum(1 for r in res_a[n] if r["integral"] < threshold)
        r, l, h = collapse_rate_wilson(collapsed_a, total_a)
        rates_a.append(r); lo_a.append(l); hi_a.append(h)
        E_rel_a.append([r["E_rel"] for r in res_a[n] if r["integral"] >= threshold])

        if res_b:
            total_b = len(res_b[n])
            collapsed_b = sum(1 for r in res_b[n] if r["integral"] < threshold)
            r, l, h = collapse_rate_wilson(collapsed_b, total_b)
            rates_b.append(r); lo_b.append(l); hi_b.append(h)
            E_rel_b.append([r["E_rel"] for r in res_b[n] if r["integral"] >= threshold])

    # --- Figura 1: Collapse rate con IC ---
    fig, ax = plt.subplots(figsize=(12, 5))
    rates_a_pct = [r*100 if r is not None else 0 for r in rates_a]
    lo_a_pct    = [l*100 if l is not None else 0 for l in lo_a]
    hi_a_pct    = [h*100 if h is not None else 0 for h in hi_a]
    ax.plot(modes, rates_a_pct, 'b-o', label=cfg_a["experiment_name"])
    ax.fill_between(modes, lo_a_pct, hi_a_pct, alpha=0.2, color='blue', label="95% CI A1")
    if res_b:
        rates_b_pct = [r*100 if r is not None else 0 for r in rates_b]
        lo_b_pct    = [l*100 if l is not None else 0 for l in lo_b]
        hi_b_pct    = [h*100 if h is not None else 0 for h in hi_b]
        ax.plot(modes, rates_b_pct, 'r-s', label=cfg_b["experiment_name"])
        ax.fill_between(modes, lo_b_pct, hi_b_pct, alpha=0.2, color='red', label="95% CI A2")
    ax.set_xlabel("Mode n"); ax.set_ylabel("Collapse rate (%)")
    ax.set_title("Collapse rate with 95% Wilson confidence intervals — Rayleigh vs Paper 1")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "collapse_rate_ci.png"), dpi=150)
    plt.close()
    print("✅ collapse_rate_ci.png")

    # --- Figura 2: E_rel mediana ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, n in enumerate(modes):
        if E_rel_a[i]:
            ax.scatter([n]*len(E_rel_a[i]), E_rel_a[i], color='blue', alpha=0.5, s=20)
    medians_a = [np.median(v) if v else np.nan for v in E_rel_a]
    ax.plot(modes, medians_a, 'b-o', label=f"Median {cfg_a['experiment_name']}")
    if res_b:
        for i, n in enumerate(modes):
            if E_rel_b[i]:
                ax.scatter([n]*len(E_rel_b[i]), E_rel_b[i], color='red', alpha=0.5, s=20)
        medians_b = [np.median(v) if v else np.nan for v in E_rel_b]
        ax.plot(modes, medians_b, 'r-s', label=f"Median {cfg_b['experiment_name']}")
    ax.set_yscale("log")
    ax.set_xlabel("Mode n"); ax.set_ylabel("E_rel")
    ax.set_title("Energy relative error — converged runs only")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "E_rel_scatter.png"), dpi=150)
    plt.close()
    print("✅ E_rel_scatter.png")

    # --- Resumen por consola ---
    print(f"\n{'n':>4} {'rA1':>6} {'CI_A1':>16} {'conv_A1':>8}", end="")
    if res_b:
        print(f" {'rA2':>6} {'CI_A2':>16} {'conv_A2':>8}", end="")
    print()
    for i, n in enumerate(modes):
        print(f"{n:>4} {rates_a[i]*100:>5.0f}% [{lo_a[i]*100:>4.0f},{hi_a[i]*100:>4.0f}]% {len(E_rel_a[i]):>8}", end="")
        if res_b:
            print(f" {rates_b[i]*100:>5.0f}% [{lo_b[i]*100:>4.0f},{hi_b[i]*100:>4.0f}]% {len(E_rel_b[i]):>8}", end="")
        print()

    # --- JSON resumen ---
    summary = {}
    for i, n in enumerate(modes):
        summary[str(n)] = {
            "A1": {
                "collapse_rate": rates_a[i],
                "ci_lo": lo_a[i],
                "ci_hi": hi_a[i],
                "n_converged": len(E_rel_a[i]),
                "E_rel_median": float(np.median(E_rel_a[i])) if E_rel_a[i] else None,
            }
        }
        if res_b:
            summary[str(n)]["A2"] = {
                "collapse_rate": rates_b[i],
                "ci_lo": lo_b[i],
                "ci_hi": hi_b[i],
                "n_converged": len(E_rel_b[i]),
                "E_rel_median": float(np.median(E_rel_b[i])) if E_rel_b[i] else None,
            }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Análisis completo → {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_a", type=str, required=True)
    parser.add_argument("--config_b", type=str, default=None)
    parser.add_argument("--n_min",    type=int, default=1)
    parser.add_argument("--n_max",    type=int, default=30)
    parser.add_argument("--seeds",    type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--out_dir",  type=str, default="outputs/analysis")
    args = parser.parse_args()

    def load_cfg(p):
        with open(p) as f:
            return yaml.safe_load(f)

    cfg_a = load_cfg(args.config_a)
    cfg_b = load_cfg(args.config_b) if args.config_b else None

    analyze(cfg_a, cfg_b,
            seeds=args.seeds,
            n_min=args.n_min,
            n_max=args.n_max,
            out_dir=args.out_dir)
