#!/usr/bin/env python3
import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

try:
    from transform_reactions import transform_reaction
except ModuleNotFoundError:
    transform_reaction = None


PREDICTION_COLUMNS = ["0th", "1th", "2th", "3th", "4th"]


def canonicalize(smiles, strip_stereo=False):
    if pd.isna(smiles):
        return ""

    smiles = str(smiles).replace(" ", "").strip().rstrip(".")
    if not smiles:
        return ""

    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        if strip_stereo:
            Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=not strip_stereo)
    except Exception:
        return smiles


def load_test_data(path):
    df = pd.read_csv(path)

    if {"REACTANT", "PRODUCT"}.issubset(df.columns):
        test = df.copy()
        for col in ["REACTANT", "REAGENT"]:
            if col not in test.columns:
                test[col] = " "
            test[col] = test[col].fillna(" ")
        test["input"] = "REACTANT:" + test["REACTANT"] + "REAGENT:" + test["REAGENT"]
        return test[["input", "PRODUCT"]].copy()

    if "updated_reaction" not in df.columns:
        raise ValueError(
            "Test CSV must contain either REACTANT/PRODUCT columns or updated_reaction."
        )
    if transform_reaction is None:
        raise ImportError(
            "Could not import transform_reactions.py. Use ../dataformat/test.csv, "
            "or run this script from a checkout that contains transform_reactions.py."
        )

    rows = []
    for row in df.itertuples(index=False):
        reactant, reagent, product = transform_reaction(row.updated_reaction)
        rows.append({"REACTANT": reactant, "REAGENT": reagent, "PRODUCT": product})

    test = pd.DataFrame(rows)
    test["input"] = "REACTANT:" + test["REACTANT"] + "REAGENT:" + test["REAGENT"].fillna(" ")
    return test[["input", "PRODUCT"]]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ReactionT5 top-k product predictions."
    )
    parser.add_argument("--test_csv", required=True, help="Path to test.csv.")
    parser.add_argument(
        "--pred_csv", required=True, help="Path to prediction output.csv."
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=5,
        help="Evaluate top-k up to this value.",
    )
    parser.add_argument(
        "--strip_stereo",
        action="store_true",
        help="Ignore stereochemistry during comparison.",
    )
    parser.add_argument(
        "--wrong_csv",
        default=None,
        help="Optional path to save wrong top-k examples.",
    )
    args = parser.parse_args()

    test = load_test_data(args.test_csv)
    pred = pd.read_csv(args.pred_csv)

    pred_cols = [col for col in PREDICTION_COLUMNS[: args.max_k] if col in pred.columns]
    if not pred_cols:
        raise ValueError(f"No prediction columns found in {args.pred_csv}.")

    if len(test) != len(pred):
        print(f"WARNING: row count differs: test={len(test)} pred={len(pred)}")

    n = min(len(test), len(pred))
    test = test.iloc[:n].reset_index(drop=True)
    pred = pred.iloc[:n].reset_index(drop=True)

    if "input" in pred.columns:
        input_matches = (test["input"].astype(str) == pred["input"].astype(str)).sum()
        if input_matches != n:
            print(f"WARNING: input strings match for only {input_matches}/{n} rows.")

    true_products = [
        canonicalize(value, strip_stereo=args.strip_stereo) for value in test["PRODUCT"]
    ]

    topk_correct = {k: 0 for k in range(1, len(pred_cols) + 1)}
    invalid_top1 = 0
    wrong_rows = []

    for i in range(n):
        predictions = [
            canonicalize(pred.loc[i, col], strip_stereo=args.strip_stereo)
            for col in pred_cols
        ]
        target = true_products[i]

        if not predictions[0]:
            invalid_top1 += 1

        for k in topk_correct:
            if target in predictions[:k]:
                topk_correct[k] += 1

        if target not in predictions:
            wrong_rows.append(
                {
                    "row": i,
                    "input": pred.loc[i, "input"] if "input" in pred.columns else test.loc[i, "input"],
                    "target": target,
                    **{f"pred_{j}": value for j, value in enumerate(predictions)},
                }
            )

    print(f"Rows evaluated: {n}")
    print(f"Prediction columns: {', '.join(pred_cols)}")
    print(f"Strip stereo: {args.strip_stereo}")
    print()

    for k, correct in topk_correct.items():
        print(f"Top-{k} accuracy: {correct / n:.4f} ({correct}/{n})")

    print()
    print(f"Wrong top-{len(pred_cols)}: {len(wrong_rows)}/{n}")
    print(f"Empty/invalid top-1 after normalization: {invalid_top1}/{n}")

    if args.wrong_csv:
        pd.DataFrame(wrong_rows).to_csv(args.wrong_csv, index=False)
        print(f"Saved wrong examples to: {args.wrong_csv}")


if __name__ == "__main__":
    main()
