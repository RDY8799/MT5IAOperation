from __future__ import annotations

import argparse

from .phase10_runner import run_phase10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 11 runner (M5 quality>quantity).")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phase10(
        symbol=args.symbol,
        entry_tf="M5",
        gate_tf="M30",
        windows=args.windows,
        seed=args.seed,
        threshold=0.55,
        min_signal_margin=0.15,
        output_prefix="phase11",
    )


if __name__ == "__main__":
    main()

