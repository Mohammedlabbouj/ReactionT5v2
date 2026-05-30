import argparse
import csv
import re
from pathlib import Path

from rdkit import Chem, RDLogger


MAP_RE = re.compile(r":(\d+)(?=\])")
CHIRAL_RE = re.compile(r"@+")

RDLogger.DisableLog("rdApp.*")


def atom_maps(smiles: str) -> list[int]:
    return [int(match) for match in MAP_RE.findall(smiles)]


def is_reagent(molecule: str) -> bool:
    maps = atom_maps(molecule)
    if not maps:
        return True
    return all(map_num >= 300 for map_num in maps)


def strip_maps_and_stereo(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return CHIRAL_RE.sub("", MAP_RE.sub("", smiles))

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def clean_side(molecules: list[str]) -> str:
    return ".".join(strip_maps_and_stereo(molecule) for molecule in molecules if molecule)


def valid_side(smiles: str) -> bool:
    return all(Chem.MolFromSmiles(molecule) is not None for molecule in smiles.split(".") if molecule)


def transform_reaction(updated_reaction: str) -> tuple[str, str, str]:
    reactant_side, product_side = updated_reaction.split(">>", 1)
    left_molecules = [molecule for molecule in reactant_side.split(".") if molecule]
    product_molecules = [molecule for molecule in product_side.split(".") if molecule]

    reactants = [molecule for molecule in left_molecules if not is_reagent(molecule)]
    reagents = [molecule for molecule in left_molecules if is_reagent(molecule)]

    return clean_side(reactants), clean_side(reagents), clean_side(product_molecules)


def transform_original_reaction(original_reaction: str) -> tuple[str, str]:
    reactant_side, product_side = original_reaction.split(">>", 1)
    reactants = clean_side([molecule for molecule in reactant_side.split(".") if molecule])
    products = clean_side([molecule for molecule in product_side.split(".") if molecule])
    return reactants, products


def transform_file(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with input_path.open(newline="", encoding="utf-8") as input_file, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        reader = csv.DictReader(input_file)
        if "updated_reaction" not in (reader.fieldnames or []):
            raise ValueError(f"{input_path} does not contain an updated_reaction column")

        writer = csv.DictWriter(output_file, fieldnames=["REACTANT", "REAGENT", "PRODUCT"])
        writer.writeheader()

        for row in reader:
            reactant, reagent, product = transform_reaction(row["updated_reaction"])
            if (not valid_side(reactant) or not valid_side(product)) and row.get(
                "original_reactions"
            ):
                reactant, product = transform_original_reaction(row["original_reactions"])
            writer.writerow({"REACTANT": reactant, "REAGENT": reagent, "PRODUCT": product})
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform mapped USPTO reactions into REACTANT,REAGENT,PRODUCT CSV files."
    )
    parser.add_argument("--input-dir", default="31k_uspto", type=Path)
    parser.add_argument("--output-dir", default="dataformat", type=Path)
    parser.add_argument("--files", nargs="+", default=["train.csv", "val.csv", "test.csv"])
    args = parser.parse_args()

    for file_name in args.files:
        input_path = args.input_dir / file_name
        output_path = args.output_dir / file_name
        count = transform_file(input_path, output_path)
        print(f"Wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
