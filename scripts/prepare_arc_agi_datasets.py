#!/usr/bin/env python3
"""Normalize local ARC-AGI-1/2 data and freeze clean experiment manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.arc_dataset_preparation import prepare_arc_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arc1_data_dir",
        default=str(PROJECT_ROOT / "agi_1" / "ARC-AGI" / "data"),
        help="Directory containing ARC-AGI-1 training/ and evaluation/ task files.",
    )
    parser.add_argument(
        "--arc2_problems_dir",
        default=str(PROJECT_ROOT / "problems"),
        help="Directory containing aggregate arc-agi_*_challenges/solutions JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "datasets" / "arc_agi"),
        help="New write-once normalized dataset directory.",
    )
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--global_train_tasks", type=int, default=320)
    parser.add_argument("--development_tasks", type=int, default=40)
    parser.add_argument("--final_heldout_tasks", type=int, default=40)
    args = parser.parse_args()

    manifest = prepare_arc_datasets(
        args.arc1_data_dir,
        args.arc2_problems_dir,
        args.output_dir,
        split_seed=args.split_seed,
        global_train_count=args.global_train_tasks,
        development_count=args.development_tasks,
        final_heldout_count=args.final_heldout_tasks,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

