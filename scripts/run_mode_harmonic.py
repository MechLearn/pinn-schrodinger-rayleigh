# scripts/run_mode_harmonic.py
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pinn1d.Harmonic.config import load_config
from src.pinn1d.Harmonic.train import run_one_mode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n",      type=int, required=True)
    parser.add_argument("--seed",   type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed

    result = run_one_mode(args.n, cfg)
    print(f"\n✅ n={result['n']} | E={result['E_learned']:.6f} | E_exact={result['E_exact']:.4f} | E_rel={result['E_rel']:.2e} | L2={result['L2']:.2e} | device={result['device']}")

if __name__ == "__main__":
    main()
