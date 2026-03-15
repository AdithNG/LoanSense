"""Generate sample loan data CSV for training."""

import argparse
from pathlib import Path

from src.data.load import load_sample_data

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "loan_data.csv"


def main():
    p = argparse.ArgumentParser(description="Generate sample loan_data.csv")
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=str(DEFAULT_PATH))
    args = p.parse_args()
    df = load_sample_data(n=args.rows, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
