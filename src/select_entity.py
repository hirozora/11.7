"""Choice of the ridge scale c per entity variant by validation NLL (docs/MODEL_PLAN.md, implementation note 18: c
multiplies B0's cross-validated ridge strengths and is chosen on the validation period from three values).

Reads <out>/run.json of every job in the --jobs files (the --out argument of each job); writes --out (json) with, per
variant, the validation NLL, training objective, largest gradient component and stopping reason of each c, the chosen
c and its scores file.
"""
import argparse

from common import ROOT, load
from model_data import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    variants = {}
    for job in [j for path in args.jobs for j in load(ROOT / path)]:
        a = job["args"]
        out = a[a.index("--out") + 1]
        run = load(ROOT / out / "run.json")
        variants.setdefault(run["model"], []).append({k: run[k] for k in ("ridge_scale", "validation_nll", "objective",
                                                                          "train_nll", "gradient_max", "stopped_by")}
                                                     | {"scores": f"{out}/scores.npz"})
    chosen = {}
    for model, runs in variants.items():
        best = min(runs, key=lambda r: r["validation_nll"])
        chosen[model] = {"ridge_scale": best["ridge_scale"], "validation_nll": best["validation_nll"], "scores": best["scores"],
                         "runs": sorted(runs, key=lambda r: r["ridge_scale"])}
        print(model, "c", round(best["ridge_scale"], 4), round(best["validation_nll"], 5),
              {round(r["ridge_scale"], 4): (round(r["validation_nll"], 5), r["stopped_by"]) for r in runs}, flush=True)
    write_json(ROOT / args.out, chosen)


if __name__ == "__main__":
    main()
