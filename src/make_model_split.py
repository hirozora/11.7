"""Explicit training and validation match lists of the who-acts model (docs/MODEL_PLAN.md, section 1).

Within each league the non-test matches are ordered by kickoff; the last N form the validation period, N being the size
of the league's test list (read only for its length). Every match sharing the cut kickoff goes to validation, so that
no training match kicks off at the same time as a validation match; the validation period can then hold slightly more
than N matches. Writes configs/manifests/<league>_model_train.json and <league>_model_validation.json.
"""
import argparse
import json

from common import ROOT, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", nargs="+", required=True)
    args = parser.parse_args()
    for league in args.leagues:
        nontest = load(ROOT / "configs" / "manifests" / f"{league}_nontest.json")
        n_val = len(load(ROOT / "configs" / "manifests" / f"{league}_test.json"))
        rows = sorted(nontest, key=lambda r: (r["kickoff"], r["match_id"]))
        cut = rows[len(rows) - n_val]["kickoff"]
        split = {"model_train": [r for r in rows if r["kickoff"] < cut], "model_validation": [r for r in rows if r["kickoff"] >= cut]}
        for name, part in split.items():
            out = [{"match_id": r["match_id"], "kickoff": r["kickoff"], "league": league, "split": name} for r in part]
            with open(ROOT / "configs" / "manifests" / f"{league}_{name}.json", "w", encoding="utf-8") as f:
                json.dump(out, f, indent=1)
        print(f"{league}: training {len(split['model_train'])}, validation {len(split['model_validation'])} "
              f"(test list {n_val}), cut kickoff {cut}", flush=True)


if __name__ == "__main__":
    main()
