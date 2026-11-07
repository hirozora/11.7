"""Entity and cell indices of the who-acts entity models (docs/MODEL_PLAN.md, section 3, second stage).

Per target (cells of the restricted terms): zone (0-8), steps (0-4), phase (0-2), opponent's last event in the five
levels of the confirmatory analyses (pass, duel, interception, clearance, other; -1 when steps > 1), league.
Per (target, slot): player, slot (team x prematch role, role 0 = unknown), team, prematch role, team style type
(league x 3 + K-means type from the league's training-period matches, src/entity_types.py team_types), left foot,
standardised height and weight (0 when missing, with missing flags).
All indices come from the input arrays of src/model_inputs.py and the model view; the team types use the training-period
action table only. Writes cfg["out"] (npz) and its .json description.
"""
import argparse

import numpy as np
import pandas as pd

from common import ROOT, build, load
from entity_types import team_types
from model_data import load_meta, scope_dirs, write_json
from model_inputs import VOCAB

OPP5 = {"pass": 0, "duel": 1, "interception": 2, "clearance": 3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    Y, meta = load_meta(cfg["scopes"])
    dirs = scope_dirs(cfg["scopes"])
    cat = {k: [] for k in ("cand_player", "cand_cat", "cand_mask", "ctx_cat", "ctx_num")}
    for d in dirs:
        A = np.load(d / "arrays_in.npz")
        for k in cat:
            cat[k].append(A[k])
    arr = {k: np.concatenate(v) for k, v in cat.items()}
    opp_names = VOCAB["opp_last_event"]
    opp5 = np.array([OPP5.get(opp_names[c], 4) for c in arr["ctx_cat"][:, 3]])
    steps = np.rint(arr["ctx_num"][:, 2] * 4).astype(int)
    cells = {"zone": arr["ctx_cat"][:, 0].astype(int), "steps": steps, "phase": arr["ctx_cat"][:, 1].astype(int),
             "opp5": np.where(steps <= 1, opp5, -1), "league": meta["league"].to_numpy()}

    mask = arr["cand_mask"]
    player_ids = np.unique(arr["cand_player"][mask])
    player = np.where(mask, np.searchsorted(player_ids, arr["cand_player"]), -1)
    role = np.where(mask, arr["cand_cat"][..., 0].astype(int), -1)  # prematch role + 1, 0 = unknown
    team = np.repeat(meta["team_id"].to_numpy()[:, None], mask.shape[1], 1)
    team_ids = np.unique(meta["team_id"])
    team_i = np.where(mask, np.searchsorted(team_ids, team), -1)
    n_roles = int(role.max()) + 1
    slot_key = team_i * n_roles + role
    slot_ids = np.unique(slot_key[mask])
    slot = np.where(mask, np.searchsorted(slot_ids, slot_key), -1)

    # team style types from each league's training-period actions
    type_of = {}
    for li, path in enumerate(cfg["scopes"]):
        data_cfg = load(ROOT / load(ROOT / path)["data_config"])
        A, _ = build(data_cfg)
        train = set(meta.loc[(meta["league"] == li) & (meta["block"] == 0), "match_id"])
        _, types = team_types(A[A["match_id"].isin(train)], [cfg["team_type_k"]], cfg["seed"])
        for t, ty in types[cfg["team_type_k"]].items():
            type_of[t] = li * cfg["team_type_k"] + int(ty)
    team_type = np.where(mask, pd.Series(team.reshape(-1)).map(type_of).fillna(-1).to_numpy().reshape(mask.shape), -1).astype(int)
    if (team_type[mask] < 0).any():
        raise ValueError("a team without a style type")

    players = {p["wyId"]: p for p in load(ROOT / "data" / "raw" / "entities" / "players.json")}
    foot = np.array([players[p]["foot"] == "left" for p in player_ids], dtype=np.float32)
    h = np.array([players[p]["height"] for p in player_ids], dtype=float)
    w = np.array([players[p]["weight"] for p in player_ids], dtype=float)
    hz = np.where(h > 0, (h - h[h > 0].mean()) / h[h > 0].std(), 0.0)
    wz = np.where(w > 0, (w - w[w > 0].mean()) / w[w > 0].std(), 0.0)
    out = {**cells, "player": player, "slot": slot, "team": team_i, "role": role, "team_type": team_type,
           "player_ids": player_ids, "slot_ids": slot_ids, "team_ids": team_ids, "left_foot": foot, "height_z": hz,
           "weight_z": wz, "height_missing": (h <= 0).astype(np.float32), "weight_missing": (w <= 0).astype(np.float32)}
    np.savez(ROOT / cfg["out"], **out)
    info = {"players": int(len(player_ids)), "slots": int(len(slot_ids)), "teams": int(len(team_ids)), "roles": n_roles,
            "team_types": int(len(cfg["scopes"]) * cfg["team_type_k"]), "targets": int(len(meta)),
            "steps_0_1_targets": int((steps <= 1).sum())}
    write_json(ROOT / (cfg["out"][:-4] + ".json"), info)
    print(info, flush=True)


if __name__ == "__main__":
    main()
