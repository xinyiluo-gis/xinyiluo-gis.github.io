"""Reproducibly split a CSV table into two subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_csv(
    input_path: Path,
    output_dir: Path | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[Path, Path]:
    """Split a CSV and return the paths of the train and test files."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    table = pd.read_csv(input_path)
    train, test = train_test_split(
        table,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    destination = output_dir or input_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / f"{input_path.stem}_train.csv"
    test_path = destination / f"{input_path.stem}_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_file, test_file = split_csv(
        args.input,
        args.output_dir,
        args.test_size,
        args.random_state,
    )
    print(f"Train file: {train_file}")
    print(f"Test file:  {test_file}")


