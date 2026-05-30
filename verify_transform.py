import argparse
import csv
import re
from pathlib import Path

from rdkit import Chem, RDLogger


MAP_RE = re.compile(r":\d+(?=\])")
CHIRAL_RE = re.compile(r"@")
RDLogger.DisableLog("rdApp.*")


def verify_file(path: Path) -> tuple[int, dict[str, int], list[str]]:
    problems: list[str] = []
    molecule_counts = {"REACTANT": 0, "REAGENT": 0, "PRODUCT": 0}
    count = 0

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        expected = ["REACTANT", "REAGENT", "PRODUCT"]
        if reader.fieldnames != expected:
            problems.append(f"expected header {expected}, got {reader.fieldnames}")
            return count, molecule_counts, problems

        for line_number, row in enumerate(reader, start=2):
            count += 1
            for column in expected:
                value = row[column]
                if MAP_RE.search(value):
                    problems.append(f"line {line_number}: atom map remains in {column}")
                if CHIRAL_RE.search(value):
                    problems.append(f"line {line_number}: chirality marker remains in {column}")
                for molecule in [part for part in value.split(".") if part]:
                    molecule_counts[column] += 1
                    if Chem.MolFromSmiles(molecule) is None:
                        problems.append(
                            f"line {line_number}: invalid SMILES in {column}: {molecule}"
                        )
            if not row["REACTANT"]:
                problems.append(f"line {line_number}: empty REACTANT")
            if not row["PRODUCT"]:
                problems.append(f"line {line_number}: empty PRODUCT")

    return count, molecule_counts, problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify transformed reaction CSV files.")
    parser.add_argument("--data-dir", default="dataformat", type=Path)
    parser.add_argument("--files", nargs="+", default=["train.csv", "val.csv", "test.csv"])
    args = parser.parse_args()

    failed = False
    total_rows = 0
    total_molecules = {"REACTANT": 0, "REAGENT": 0, "PRODUCT": 0}
    for file_name in args.files:
        path = args.data_dir / file_name
        count, molecule_counts, problems = verify_file(path)
        total_rows += count
        for column, molecule_count in molecule_counts.items():
            total_molecules[column] += molecule_count

        count_text = ", ".join(
            f"{column} molecules={molecule_counts[column]}" for column in molecule_counts
        )
        if problems:
            failed = True
            print(f"{path}: FAILED ({count} rows, {count_text})")
            for problem in problems[:20]:
                print(f"  - {problem}")
            if len(problems) > 20:
                print(f"  - ... {len(problems) - 20} more problems")
        else:
            print(f"{path}: OK ({count} rows, {count_text})")

    total_text = ", ".join(
        f"{column} molecules={total_molecules[column]}" for column in total_molecules
    )
    print(f"TOTAL: {total_rows} rows, {total_text}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
