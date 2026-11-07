"""Causality checks of the history statistics and graphs (src/history_stats.py). Stops at the first failure.
  1  who acted in every validation match is replaced by a random candidate: no feature, node or edge changes
  2  who acted in training matches on or after a sampled day is replaced by a random candidate: every target whose
     snapshot lies on or before that day keeps its features and its snapshot's graph; later targets change
  3  the candidates of every validation target are replaced by players that occur nowhere else: no feature, node or
     edge changes (a snapshot's graph never depends on the targets that use it)
  4  the candidates of training targets on or after a sampled day are replaced by such players: every snapshot on or
     before that day keeps its nodes, features and edges, and the earlier targets keep their candidates' node rows;
     positive control: the snapshots after that day change
"""
import argparse

import numpy as np

from common import ROOT, load
from history_stats import compute, target_days
from model_data import load_meta


def candidate_features(H, rows):
    """(rows, C, 88): node features of each candidate's player and slot, zero without a node or for padding."""
    out = np.zeros(H["player_row"][rows].shape + (2 * H["node_feat"].shape[1],), np.float32)
    for j, key in enumerate(("player_row", "slot_row")):
        r = H[key][rows]
        has = r >= 0
        out[..., j * H["node_feat"].shape[1]:(j + 1) * H["node_feat"].shape[1]][has] = H["node_feat"][r[has]]
    return out


def snapshot_content(H, snapshots):
    """Nodes (snapshot, kind, id, features) and edges (snapshot, type, source kind and id, destination kind and id,
    weight) of the given snapshots, sorted, without row numbers."""
    keep = np.isin(H["node_snapshot"], snapshots)
    nodes = np.column_stack([H["node_snapshot"][keep], H["node_kind"][keep], H["node_id"][keep], H["node_feat"][keep]])
    nodes = nodes[np.lexsort(nodes[:, :3].T[::-1])]
    e = H["edges"]
    src, dst = e[:, 1].astype(int), e[:, 2].astype(int)
    keep_e = np.isin(H["node_snapshot"][src], snapshots)
    src, dst = src[keep_e], dst[keep_e]
    edges = np.column_stack([H["node_snapshot"][src], e[keep_e, 0], H["node_kind"][src], H["node_id"][src],
                             H["node_kind"][dst], H["node_id"][dst], e[keep_e, 3]])
    edges = edges[np.lexsort(edges[:, :6].T[::-1])]
    return nodes, edges


def same_content(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def node_keys(H, rows):
    """(kind, id) of each candidate's player and slot node, -1 without a node."""
    out = []
    for key in ("player_row", "slot_row"):
        r = H[key][rows]
        has = r >= 0
        out.append(np.where(has, H["node_id"][np.maximum(r, 0)], -1))
    return np.stack(out, -1)


def fresh_players(E, rows):
    """A copy of E in which every candidate of the given targets is replaced by a new player that occurs nowhere else
    (one new player per original player, so that the new players build up history of their own)."""
    F = dict(E)
    F["player"] = E["player"].copy()
    block = F["player"][rows]
    real = block >= 0
    block[real] += E["player"].max() + 1
    F["player"][rows] = block
    return F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    rng = np.random.default_rng(args.seed)
    Y, meta = load_meta(cfg["scopes"])
    E = dict(np.load(ROOT / cfg["terms"]))
    day = target_days(cfg, meta)
    block = meta["block"].to_numpy()
    run = lambda E_, actor: compute(E_, actor, meta, day, cfg["kappa"], cfg["knn"])
    base = run(E, Y["actor_slot"])
    all_rows = np.arange(len(meta))
    all_snaps = np.arange(len(base["snapshot_day"]))
    n_cand = (E["player"] >= 0).sum(1)
    train_days = np.unique(day[block == 0])
    sampled = rng.choice(train_days[5:-5], size=3, replace=False)

    def fail(msg):
        raise SystemExit(f"check failed: {msg}")

    # 1: validation actors
    actor = Y["actor_slot"].copy()
    val = block == 1
    actor[val] = rng.integers(0, n_cand[val])
    other = run(E, actor)
    ok = (np.array_equal(candidate_features(base, all_rows), candidate_features(other, all_rows))
          and same_content(snapshot_content(base, all_snaps), snapshot_content(other, all_snaps)))
    print("1 validation actors replaced: features and graphs unchanged:", ok, flush=True)
    if not ok:
        fail("validation matches enter the history")
    # 2: training actors from a sampled day on
    for d in sampled:
        actor = Y["actor_slot"].copy()
        late = (block == 0) & (day >= d)
        actor[late] = rng.integers(0, n_cand[late])
        other = run(E, actor)
        early = np.flatnonzero((block == 0) & (day <= d))
        later = np.flatnonzero((block == 0) & (day > d))
        early_snaps = np.unique(base["target_snapshot"][early])
        same_early = np.array_equal(candidate_features(base, early), candidate_features(other, early))
        same_graph = same_content(snapshot_content(base, early_snaps), snapshot_content(other, early_snaps))
        changed_later = not np.array_equal(candidate_features(base, later), candidate_features(other, later))
        print(f"2 day {d}: earlier targets' features unchanged {same_early}, their graphs unchanged {same_graph}, "
              f"later targets changed {changed_later}", flush=True)
        if not (same_early and same_graph and changed_later):
            fail(f"actors at day {d}")
    # 3: validation candidates
    other = run(fresh_players(E, np.flatnonzero(val)), Y["actor_slot"])
    ok = same_content(snapshot_content(base, all_snaps), snapshot_content(other, all_snaps))
    keep = np.flatnonzero(~val)
    ok_rows = np.array_equal(node_keys(base, keep), node_keys(other, keep))
    new_rows = node_keys(other, np.flatnonzero(val))[..., 0][E["player"][val] >= 0]
    print(f"3 validation candidates replaced: graphs unchanged {ok}, other targets' node rows unchanged {ok_rows}, "
          f"replaced candidates without a player node {bool((new_rows == -1).all())}", flush=True)
    if not (ok and ok_rows and (new_rows == -1).all()):
        fail("the validation snapshot depends on the validation targets")
    # 4: training candidates from a sampled day on
    for d in sampled:
        late = np.flatnonzero((block == 0) & (day >= d))
        other = run(fresh_players(E, late), Y["actor_slot"])
        snap_day = base["snapshot_day"]
        upto = np.flatnonzero(snap_day <= d)
        after = np.flatnonzero((snap_day > d) & (snap_day < np.iinfo(np.int64).max))
        same_upto = same_content(snapshot_content(base, upto), snapshot_content(other, upto))
        early = np.flatnonzero((block == 0) & (day < d))
        same_rows = np.array_equal(node_keys(base, early), node_keys(other, early))
        changed_after = not same_content(snapshot_content(base, after), snapshot_content(other, after))
        print(f"4 day {d}: snapshots up to the day unchanged {same_upto}, earlier targets' node rows unchanged {same_rows}, "
              f"later snapshots changed {changed_after}", flush=True)
        if not (same_upto and same_rows and changed_after):
            fail(f"candidates at day {d}")
    print("history causality checks passed", flush=True)


if __name__ == "__main__":
    main()
