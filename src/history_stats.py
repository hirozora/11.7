"""History statistics and history graphs per snapshot, for M3, M2+3 and M4 (docs/MODEL_PLAN.md, section 2, items 5-6).

Cells: zone (9), steps (5), phase (3), opponent event at steps 0-1 (5) = 22. Each candidate row of a training match
gives one opportunity in each cell of its target (the opponent cell only at steps 0-1) and one participation when the
candidate is the actor. A snapshot of league l and UTC day d sums the training matches of l on days before d; the
validation targets of l use the frozen snapshot that sums every training match of l.
A snapshot's graph is a function of that history alone: its nodes are the players and slots (team x prematch role) with
at least one opportunity in it, their teams, their prematch roles and the league. The targets of a snapshot never add
nodes or edges, so a candidate without history (or whose slot has none) has no node; the models then give that part
of his representation zero (src/entity_models.py).
Node features (44) of players and slots: logit of the rate shrunk with pseudo-count kappa toward the parent minus logit
of the parent (22), and log1p(opportunities) (22); other nodes have zero features. Parent of a slot: the league's rate
for its prematch role in the snapshot; parent of a player: that rate for the prematch role he held most often in the
snapshot's history.
Edge types (weight): 0 player-slot (history opportunities of the player in the slot), 1 slot-team (1), 2 slot-role (1),
3 player-role, the parent role (1), 4 team-league (1), 5 player-player, the k nearest neighbours by cosine similarity
of the player features (1).
Writes cfg["out"] (npz): node_feat, node_kind, node_id, node_snapshot, edges (type, source row, destination row,
weight), and per (target, slot) the node rows of the candidate's player and slot (-1 when he or the slot has no node,
and for padding) and the target's snapshot.
"""
import argparse

import numpy as np
import pandas as pd

from common import ROOT, load
from model_data import load_meta, write_json

N_CELLS = 22
KINDS = {"player": 0, "slot": 1, "team": 2, "role": 3, "league": 4}


def cell_matrix(E, rows_t):
    """(rows, 22) 0/1 opportunity indicators of each candidate row's target cells."""
    n = len(rows_t)
    M = np.zeros((n, N_CELLS), np.float32)
    r = np.arange(n)
    M[r, E["zone"][rows_t]] = 1
    M[r, 9 + E["steps"][rows_t]] = 1
    M[r, 14 + E["phase"][rows_t]] = 1
    opp = E["opp5"][rows_t]
    on = opp >= 0
    M[r[on], 17 + opp[on]] = 1
    return M


def target_days(cfg, meta):
    kickoff = {}
    for path in cfg["scopes"]:
        mc = load(ROOT / path)
        for name in ("train_manifest", "validation_manifest"):
            for r in load(ROOT / mc[name]):
                kickoff[r["match_id"]] = r["kickoff"]
    return meta["match_id"].map(kickoff).to_numpy() // 86400


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def compute(E, actor, meta, day, kappa, knn):
    mask = E["player"] >= 0
    rows_t, slots = np.nonzero(mask)
    block, league = meta["block"].to_numpy(), meta["league"].to_numpy()
    n_roles = int(E["role"].max()) + 1
    player_r, slot_r, role_r = E["player"][rows_t, slots], E["slot"][rows_t, slots], E["role"][rows_t, slots]
    slot_team = E["slot_ids"] // n_roles  # slot index -> team index (slot key = team * n_roles + role)
    slot_role = E["slot_ids"] % n_roles
    train = block[rows_t] == 0
    cells = cell_matrix(E, rows_t)
    part = cells * (slots == actor[rows_t])[:, None]
    snap_day = np.where(block == 1, np.iinfo(np.int64).max, day)
    snaps = pd.DataFrame({"league": league, "day": snap_day}).drop_duplicates().sort_values(["league", "day"]).reset_index(drop=True)
    snap_index = {(l, d): i for i, (l, d) in enumerate(zip(snaps["league"], snaps["day"]))}
    target_snap = np.array([snap_index[(l, d)] for l, d in zip(league, snap_day)])
    snap_r = target_snap[rows_t]

    out = {"node_feat": [], "node_kind": [], "node_id": [], "node_snapshot": [], "edges": []}
    node_of = {}
    n_total = 0
    for li in np.unique(league):
        h = train & (league[rows_t] == li)  # history rows of the league
        days = np.sort(np.unique(day[rows_t][h]))
        d_pos = np.searchsorted(days, day[rows_t][h])
        P_ids, S_ids = np.unique(player_r[h]), np.unique(slot_r[h])
        pi, si = np.searchsorted(P_ids, player_r[h]), np.searchsorted(S_ids, slot_r[h])
        cO_p, cP_p = (np.zeros((len(days), len(P_ids), N_CELLS)) for _ in range(2))
        cO_s, cP_s = (np.zeros((len(days), len(S_ids), N_CELLS)) for _ in range(2))
        role_cnt = np.zeros((len(days), len(P_ids), n_roles))
        np.add.at(cO_p, (d_pos, pi), cells[h]); np.add.at(cP_p, (d_pos, pi), part[h])
        np.add.at(cO_s, (d_pos, si), cells[h]); np.add.at(cP_s, (d_pos, si), part[h])
        np.add.at(role_cnt, (d_pos, pi, role_r[h]), 1)
        cO_p, cP_p, cO_s, cP_s, role_cnt = (np.cumsum(a, 0) for a in (cO_p, cP_p, cO_s, cP_s, role_cnt))
        ps = pd.DataFrame({"d": d_pos, "p": player_r[h], "s": slot_r[h]}).value_counts().rename("n").reset_index()

        for (l, d), i in snap_index.items():
            if l != li:
                continue
            k = int(np.searchsorted(days, d, side="left")) - 1
            if k < 0:  # no history yet: an empty graph
                continue
            jp, js = np.flatnonzero(cO_p[k].sum(1) > 0), np.flatnonzero(cO_s[k].sum(1) > 0)
            players, slots_n = P_ids[jp], S_ids[js]
            Op, Pp, Os, Ps = cO_p[k][jp], cP_p[k][jp], cO_s[k][js], cP_s[k][js]
            s_role = slot_role[slots_n]
            role_rate = {r: (Ps[s_role == r].sum(0) + 1) / (Os[s_role == r].sum(0) + 2) for r in range(n_roles)}
            s_parent = np.stack([role_rate[r] for r in s_role])
            p_role = role_cnt[k][jp].argmax(1)
            p_parent = np.stack([role_rate[r] for r in p_role])
            f_p = np.concatenate([logit((Pp + kappa * p_parent) / (Op + kappa)) - logit(p_parent), np.log1p(Op)], 1)
            f_s = np.concatenate([logit((Ps + kappa * s_parent) / (Os + kappa)) - logit(s_parent), np.log1p(Os)], 1)
            teams = np.unique(slot_team[slots_n])
            roles = np.unique(np.concatenate([s_role, p_role]))
            groups = [("player", players, f_p), ("slot", slots_n, f_s), ("team", teams, None), ("role", roles, None),
                      ("league", np.array([li]), None)]
            row = {}
            base = n_total
            for kind, ids, f in groups:
                row[kind] = dict(zip(ids.tolist(), range(base, base + len(ids))))
                out["node_feat"].append(f.astype(np.float32) if f is not None else np.zeros((len(ids), 2 * N_CELLS), np.float32))
                out["node_kind"].append(np.full(len(ids), KINDS[kind], np.int8))
                out["node_id"].append(ids.astype(np.int64))
                out["node_snapshot"].append(np.full(len(ids), i, np.int32))
                base += len(ids)
            e = []
            for (p, s), n in ps[ps["d"] <= k].groupby(["p", "s"])["n"].sum().items():
                e.append((0, row["player"][p], row["slot"][s], float(n)))
            for s, r in zip(slots_n.tolist(), s_role.tolist()):
                e.append((1, row["slot"][s], row["team"][int(slot_team[s])], 1.0))
                e.append((2, row["slot"][s], row["role"][r], 1.0))
            for p, r in zip(players.tolist(), p_role.tolist()):
                e.append((3, row["player"][p], row["role"][r], 1.0))
            for t in teams.tolist():
                e.append((4, row["team"][t], row["league"][int(li)], 1.0))
            if len(players) > knn:
                Z = f_p / np.maximum(np.linalg.norm(f_p, axis=1, keepdims=True), 1e-8)
                sim = Z @ Z.T
                np.fill_diagonal(sim, -np.inf)
                for a, nb in zip(range(len(players)), np.argsort(-sim, 1, kind="stable")[:, :knn]):
                    for b in nb:
                        e.append((5, row["player"][int(players[a])], row["player"][int(players[b])], 1.0))
            out["edges"].append(np.array(e, dtype=np.float64).reshape(-1, 4))
            node_of.update({(i, 0, p): r for p, r in row["player"].items()})
            node_of.update({(i, 1, s): r for s, r in row["slot"].items()})
            n_total = base
    N, C = mask.shape
    p_row = np.full((N, C), -1, np.int64)
    s_row = np.full((N, C), -1, np.int64)
    p_row[rows_t, slots] = [node_of.get((i, 0, p), -1) for i, p in zip(snap_r.tolist(), player_r.tolist())]
    s_row[rows_t, slots] = [node_of.get((i, 1, s), -1) for i, s in zip(snap_r.tolist(), slot_r.tolist())]
    res = {k: np.concatenate(v) for k, v in out.items()}
    res.update({"player_row": p_row, "slot_row": s_row, "target_snapshot": target_snap,
                "snapshot_league": snaps["league"].to_numpy(), "snapshot_day": snaps["day"].to_numpy()})
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    Y, meta = load_meta(cfg["scopes"])
    E = dict(np.load(ROOT / cfg["terms"]))
    H = compute(E, Y["actor_slot"], meta, target_days(cfg, meta), cfg["kappa"], cfg["knn"])
    np.savez(ROOT / cfg["out"], **H)
    real = E["player"] >= 0
    info = {"snapshots": int(len(H["snapshot_day"])), "nodes": int(len(H["node_feat"])), "edges": int(len(H["edges"])),
            "candidate_rows_without_player_node": int((H["player_row"][real] < 0).sum()),
            "candidate_rows_without_slot_node": int((H["slot_row"][real] < 0).sum()),
            "edges_by_type": {str(t): int((H["edges"][:, 0] == t).sum()) for t in range(6)}, "kappa": cfg["kappa"],
            "knn": cfg["knn"], "features": int(H["node_feat"].shape[1])}
    write_json(ROOT / (cfg["out"][:-4] + ".json"), info)
    print(info, flush=True)


if __name__ == "__main__":
    main()
