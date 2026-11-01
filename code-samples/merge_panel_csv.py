"""Merge yearly CSV tables into one standardized panel dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PRESERVED_COLUMNS = {"x", "y", "t", "longitude", "latitude"}


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with compact, lowercase column names."""
    rename_map = {
        column: column.lower() if column.lower() in PRESERVED_COLUMNS else column[:3].lower()
        for column in frame.columns
    }
    return frame.rename(columns=rename_map)


def merge_yearly_csvs(
    base_dir: Path,
    years: list[int],
    filename: str = "combined_data_valid.csv",
) -> pd.DataFrame:
    """Load available ``csv_<year>/<filename>`` files and concatenate them."""
    tables: list[pd.DataFrame] = []

    for year in years:
        source = base_dir / f"csv_{year}" / filename
        if not source.exists():
            print(f"Skipping missing file: {source}")
            continue

        table = standardize_columns(pd.read_csv(source))
        if "t" not in table.columns:
            table["t"] = year
        tables.append(table)
        print(f"Loaded {source} ({len(table):,} rows)")

    if not tables:
        raise FileNotFoundError("No yearly CSV files were found.")

    return pd.concat(tables, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--years", type=int, nargs="+", default=[1990, 2000, 2010, 2020])
    parser.add_argument("--filename", default="combined_data_valid.csv")
    parser.add_argument("--output", type=Path, default=Path("merged_panel_data.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merged = merge_yearly_csvs(args.base_dir, args.years, args.filename)
    merged.to_csv(args.output, index=False)
    print(f"Saved {len(merged):,} rows to {args.output}")


