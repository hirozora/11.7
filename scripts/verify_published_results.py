"""Check published England summaries against their per-match likelihood contributions."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pooling import forward_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    path = ROOT / args.report_dir
    meta = json.loads((path / "metadata.json").read_text())
    contributions = pd.read_csv(path / "per_match.csv", float_precision="round_trip")
    manifest = json.loads((ROOT / meta["manifest"]).read_text())
    test = json.loads((ROOT / meta["test_manifest"]).read_text())
    allowed = {r["match_id"] for r in manifest}
    forbidden = {r["match_id"] for r in test}
    assert not allowed & forbidden
    assert set(contributions.match_id) <= allowed
    assert not set(contributions.match_id) & forbidden
    checked = 0
    for part in ("population", "ladder"):
        summary = json.loads((path / f"{part}.json").read_text())
        assert summary["league"] == meta["league"]
        items = summary["comparisons"] if part == "population" else {**summary["styles"], **summary["interactions"]}
        data = contributions[contributions.part == part]
        assert set(data.comparison) == set(items)
        rng = np.random.default_rng(meta["seed"])
        for name, item in items.items():
            rows = data[data.comparison == name]
            assert not rows.match_id.duplicated().any()
            sums = rows.log_likelihood_difference.to_numpy()
            counts = rows.rows.to_numpy()
            assert np.isfinite(sums).all() and (counts > 0).all()
            assert int(counts.sum()) == item["rows"]
            if part == "population":
                assert len(rows) == item["matches"]
            np.testing.assert_allclose(1000 * sums.sum() / counts.sum(), item["gain"], atol=1e-12, rtol=0)
            result = forward_pool([{"sum": sums, "n": counts}], meta["bootstrap"], rng)
            np.testing.assert_allclose(result["ci"], item["ci"], atol=1e-12, rtol=0)
            checked += 1
    assert set(contributions.part) == {"population", "ladder"}
    print(f"Verified {checked} comparisons, their match-bootstrap intervals and test-manifest exclusion.")


if __name__ == "__main__":
    main()
