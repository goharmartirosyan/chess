#!/usr/bin/env python3
"""Merge chess mistake CSVs without altering existing field values.

Usage:
    python3 merge_chess_csvs.py
    python3 merge_chess_csvs.py --folder /path/to/chess --output combined_chess_games.csv
"""

import argparse
import csv
import re
from pathlib import Path


ARCHIVE_MONTH = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_](0[1-9]|1[0-2])(?!\d)")
ADDED_FIELDS = ("season", "unique_game_id", "source_csv")


def detect_season(source_pgn: str) -> str:
    match = ARCHIVE_MONTH.search(source_pgn)
    return f"{match.group(1)}-{match.group(2)}" if match else ""


def open_csv(path: Path):
    # newline="" is required by Python's CSV parser; UTF-8 BOM is supported.
    return path.open("r", encoding="utf-8-sig", newline="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("."),
                        help="Folder containing the bad_moves_depth*.csv files")
    parser.add_argument("--output", type=Path, default=Path("combined_chess_games.csv"),
                        help="Output CSV path; must not already exist")
    args = parser.parse_args()

    folder = args.folder.expanduser().resolve()
    output = args.output.expanduser().resolve()
    files = sorted(
        p for p in folder.glob("bad_moves_depth*.csv")
        if p.is_file() and p.resolve() != output
    )
    if not files:
        raise SystemExit(f"No matching CSV files found in {folder}")
    if output.exists():
        raise SystemExit(f"Output already exists: {output}\nRefusing to overwrite it.")
    if not output.parent.exists():
        raise SystemExit(f"Output directory does not exist: {output.parent}")

    # First pass: collect the original columns from every file, in their
    # original order. Never replace an existing column with a new one.
    original_columns = []
    for path in files:
        with open_csv(path) as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"Missing header: {path}")
            names = reader.fieldnames
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate column names in {path.name}")
            for name in names:
                if name not in original_columns:
                    original_columns.append(name)

    # Reserve any added-field names already present in the source files.
    for col in ADDED_FIELDS:
        if col in original_columns:
            raise ValueError(
                f"Input already contains '{col}'. Refusing to overwrite data."
            )

    output_fields = original_columns + list(ADDED_FIELDS)
    total = 0
    missing_season = 0

    # Exclusive creation (x): original files and pre-existing output stay safe.
    try:
        with output.open("x", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=output_fields,
                                    extrasaction="raise")
            writer.writeheader()

            for path in files:
                count = 0
                with open_csv(path) as stream:
                    reader = csv.DictReader(stream)
                    for row in reader:
                        if None in row:  # malformed CSV: too many fields
                            raise ValueError(f"Unexpected extra fields in {path.name}")
                        source_pgn = row.get("source_pgn") or ""
                        game_index = row.get("game_index") or ""
                        season = detect_season(source_pgn)
                        if not season:
                            missing_season += 1

                        # Archive + game_index distinguishes the same numeric
                        # game index used in different months/years.
                        game_id = (
                            f"{source_pgn}::{game_index}"
                            if source_pgn and game_index else ""
                        )

                        row.update({
                            "season": season,
                            "unique_game_id": game_id,
                            "source_csv": path.name,
                        })
                        writer.writerow(row)
                        count += 1
                total += count
                print(f"{path.name}: {count:,} rows")

    except Exception:
        # Avoid leaving a partial output on disk if a source file is malformed.
        output.unlink(missing_ok=True)
        raise

    print(f"\nMerged {len(files)} files and {total:,} rows -> {output}")
    if missing_season:
        print(f"WARNING: {missing_season:,} rows had no recognizable YYYY-MM "
              "in source_pgn. Their season is empty; original values are intact.")


if __name__ == "__main__":
    main()
