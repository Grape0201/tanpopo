#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "sudachipy>=0.6.8",
# ]
# ///
"""Build a Sudachi user dictionary (.dic) from CSV files in user_dict/."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYSTEM_DIC = REPO_ROOT / "resources" / "system.dic"
DEFAULT_OUTPUT_DIC = REPO_ROOT / "resources" / "user.dic"
DEFAULT_CSV_DIR = Path(__file__).resolve().parent

EXPECTED_COLUMNS = 19


def collect_csv_files(inputs: list[Path]) -> list[Path]:
    if inputs:
        return inputs
    return sorted(DEFAULT_CSV_DIR.glob("*.csv"))


def prepare_clean_csv(csv_files: list[Path], dest_path: Path) -> int:
    """Validate rows, strip blank lines and '#' comments, and write a clean CSV."""
    row_count = 0
    with dest_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, lineterminator="\n")
        for csv_path in csv_files:
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            with csv_path.open("r", encoding="utf-8") as in_f:
                for line_no, raw_line in enumerate(in_f, start=1):
                    stripped = raw_line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    row = next(csv.reader([raw_line]))
                    if len(row) != EXPECTED_COLUMNS:
                        raise ValueError(
                            f"{csv_path}:{line_no}: expected {EXPECTED_COLUMNS} columns, got {len(row)}: {raw_line.rstrip()}"
                        )
                    try:
                        left_id = int(row[1])
                        right_id = int(row[2])
                        int(row[3])
                    except ValueError as exc:
                        raise ValueError(
                            f"{csv_path}:{line_no}: left_id, right_id, cost must be integers"
                        ) from exc
                    if left_id < 0 or right_id < 0:
                        raise ValueError(
                            f"{csv_path}:{line_no}: left_id and right_id must be >= 0 (got {left_id}, {right_id})"
                        )
                    writer.writerow(row)
                    row_count += 1
    return row_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile Sudachi user dictionary CSV(s) into a .dic binary."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help="Input CSV files (default: all *.csv in user_dict/)",
    )
    parser.add_argument(
        "-s",
        "--system-dic",
        type=Path,
        default=DEFAULT_SYSTEM_DIC,
        help=f"Path to system.dic (default: {DEFAULT_SYSTEM_DIC})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIC,
        help=f"Output .dic path (default: {DEFAULT_OUTPUT_DIC})",
    )
    parser.add_argument(
        "-d",
        "--description",
        default="tanpopo user dictionary",
        help="Description embedded in the compiled dictionary",
    )
    args = parser.parse_args()

    if not args.system_dic.exists():
        print(f"Error: system dictionary not found: {args.system_dic}", file=sys.stderr)
        return 1

    csv_files = collect_csv_files(args.csv_files)
    if not csv_files:
        print(f"No CSV files found in {DEFAULT_CSV_DIR}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        clean_csv = Path(tmp_dir) / "merged_user.csv"
        try:
            row_count = prepare_clean_csv(csv_files, clean_csv)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if row_count == 0:
            print(
                "No active entries found in CSV files (only comments/blank lines). Skipping build."
            )
            return 0

        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp_dic = Path(tmp_dir) / "user.dic"
        sudachipy_bin = Path(sys.executable).with_name("sudachipy")
        cmd = [
            str(sudachipy_bin),
            "ubuild",
            "-s",
            str(args.system_dic),
            "-o",
            str(tmp_dic),
            "-d",
            args.description,
            str(clean_csv),
        ]
        subprocess.run(cmd, check=True)
        shutil.move(str(tmp_dic), str(args.output))

    print(f"Built {args.output} ({row_count} entries) from {len(csv_files)} CSV file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
