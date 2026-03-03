from __future__ import annotations

import argparse

from .features import process_features
from .labeling_triple_barrier import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build features + triple barrier dataset.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--tf", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("PROGRESS 1/2 Gerando features", flush=True)
    feat_path = process_features(symbol=args.symbol, timeframe=args.tf)
    print("PROGRESS 2/2 Gerando dataset (triple barrier)", flush=True)
    dataset_path = build_dataset(symbol=args.symbol, timeframe=args.tf)
    print(f"features={feat_path}")
    print(f"dataset={dataset_path}")


if __name__ == "__main__":
    main()
