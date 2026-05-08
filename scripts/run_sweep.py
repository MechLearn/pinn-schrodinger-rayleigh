# scripts/run_sweep.py
import argparse
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pinn1d.InfiniteWell.config import load_config
from src.pinn1d.InfiniteWell.train import run_one_mode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--n_min",    type=int, default=1)
    parser.add_argument("--n_max",    type=int, default=20)
    parser.add_argument("--seeds",    type=int, nargs="+", default=[0,1,2,3,4])
    args = parser.parse_args()

    cfg = load_config(args.config)

    results = []
    for seed in args.seeds:
        for n in range(args.n_min, args.n_max + 1):
            cfg["seed"] = seed

            # Checkpoint — saltar si ya existe
            save_dir = os.path.join(
                cfg["save_dir"],
                f"{cfg['experiment_name']}_seed{seed}",
                f"mode_{n}"
            )
            metrics_path = os.path.join(save_dir, "metrics.json")

            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    r = json.load(f)
                print(f"⏭️  n={n} seed={seed} ya existe — E_rel={r['E_rel']:.2e} L2={r['L2']:.2e}")
                results.append(r)
                continue

            print(f"\n{'='*50}")
            print(f"Corriendo n={n} seed={seed}")
            print(f"{'='*50}")
            try:
                r = run_one_mode(n, cfg)
                results.append(r)
                print(f"✅ n={r['n']} seed={seed} | E={r['E_learned']:.6f} | E_exact={r['E_exact']:.6f} | E_rel={r['E_rel']:.2e} | L2={r['L2']:.2e}")
            except Exception as e:
                print(f"❌ n={n} seed={seed} ERROR: {e}")
                results.append({"n": n, "seed": seed, "error": str(e)})

    print(f"\n{'='*50}")
    print(f"BARRIDO COMPLETO — {len(results)} corridas")
    print(f"{'='*50}")
    for r in results:
        if "error" in r:
            print(f"  n={r['n']} seed={r['seed']} ❌ {r['error']}")
        else:
            print(f"  n={r['n']} seed={r.get('seed','?')} ✅ E_rel={r['E_rel']:.2e} L2={r['L2']:.2e}")

if __name__ == "__main__":
    main()
