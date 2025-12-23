"""
Lightweight script to verify that Weights & Biases (wandb) can be reached.

Usage:
    python utils/wandb_ping.py --project dummy-sbert --run_name test-run
"""

import argparse
import random
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal wandb connectivity test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project", default="dummy-project", help="wandb project name")
    parser.add_argument("--run_name", default="wandb-ping", help="wandb run name")
    parser.add_argument(
        "--entity", default=None, help="wandb entity/team (if required by your account)"
    )
    parser.add_argument(
        "--steps", type=int, default=5, help="Number of dummy metric steps to log"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to wait between logs (helps check streaming updates)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run wandb in offline mode (useful to test local logging)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed in this environment. "
            "Run `pip install wandb` inside your virtualenv first."
        ) from exc

    run = wandb.init(
        project=args.project,
        name=args.run_name,
        entity=args.entity,
        mode="offline" if args.offline else "online",
        config={
            "purpose": "connectivity_test",
            "random_seed": random.randint(0, 1_000_000),
        },
    )

    print(f"[wandb_ping] Started run: {run.project}/{run.name} ({run.id})")

    for step in range(1, args.steps + 1):
        metric = random.random()
        wandb.log({"dummy_metric": metric, "step": step})
        print(f"[wandb_ping] step={step} dummy_metric={metric:.4f}")
        time.sleep(args.sleep)

    wandb.finish()
    print("[wandb_ping] Completed wandb logging successfully.")


if __name__ == "__main__":
    main()
