"""Entity parts of the who-acts head on a frozen offset (docs/MODEL_PLAN.md, section 3, second stage; implementation
notes 15, 16 and 18).

score[i, k] = offset[i, k] + entity[i, k]; the offset (cross-fitted M1 scores) is fixed. Restricted terms, as in
src/fit_b0.py: slot terms (zone; steps; opponent event at steps 0-1) and player terms (zone; steps; phase; opponent
event at steps 0-1); term t adds <A_t[cell_t(i)], e_level(k)>, one table A_t per term and level.
Representations (docs/MODEL_PLAN.md: e_player = e_player_type + e_slot + r_player, e_slot = e_position_type + r_slot):
  e_slot   = [types] e_role + e_(team style type x role)  [ids] + r_slot  [hist] + h_slot
  e_player = e_slot + [types] W_physique (left foot, height, weight and their missing flags)  [ids] + r_player
             [hist] + h_player
  Team-level parts (a team residual, a team style main effect) are left out: they are the same for every candidate of
  a target and cancel in the softmax.
  hist   h = the encoder output for the candidate's player or slot node in the target's snapshot (src/history_stats.py):
         input Linear(history features) for players and slots, embeddings for teams, roles and the league, then two
         FastHGTConv layers (residual, LayerNorm) in which every node has a self-loop; zero when the player or slot
         has no node (no history in the snapshot). All snapshots are encoded together (they share no edges).
  mp     the encoder also passes messages over the history edges (player-slot, slot-team, slot-role, player-role,
         team-league; knn adds the player-player edges); an edge's weight w enters its attention score as + log w.
         Without mp every node attends only to itself, so the encoders of M2+3 and M4 differ only by the history edges.
Penalty, on the summed NLL of the reachable training targets: for every penalised vector v (rows of r_player, r_slot,
e_role and e_(type x role); columns of W_physique) and every cell z of every term t that v affects,
1/2 * c * lam * <a_t[z], v>^2. a_t[z] is v's effect on the score of cell z: A_p for player-level vectors; A_s + A_p
for slot-level vectors in terms at both levels (the player inherits the slot), A_p in the player-only phase term.
lam is B0's cross-validated strength for that level and term (src/fit_b0.py, b0.json): player_t for r_player, slot_t
(phase: player_phase) for r_slot, physique_zone for W_physique, type_steps for the type-level vectors; c
(--ridge-scale) is chosen on the validation period. The penalty is invariant to rescaling A against the vectors.
Every other parameter (A tables, encoder) gets 1/2 * weight_decay * w^2 on the mean NLL. The diagnostic d has no
A tables: its penalised vectors get 1/2 * c * lam_player_zone * |v|^2.
Variants:
  m2 = types + ids; m2_main (zone terms only); m2_no_types; m2_no_opp (no opponent-event terms)
  m3 = types + hist (no ids); m23 = types + ids + hist; m4 = types + ids + hist + mp + knn; m4_no_knn (no player-player
  edges); m4_no_id = types + hist + mp + knn
  d = unrestricted diagnostic: offset + MLP([M1 context, e_player, e_slot, target cell embeddings]) with m2's
      representations
  degenerate: the B0 problem in PyTorch (one free parameter per entity x cell and B0's type terms, B0's ridge
      strengths), float64, full-batch L-BFGS; compared with src/fit_b0.py's solution
Fitting (every variant except degenerate): least-squares distillation of B0's entity part, centred within each
target, by Adam (cfg["distill_epochs"]); then full-batch L-BFGS on the penalised objective, in rounds of
cfg["lbfgs_round_iterations"] iterations, until the largest gradient component is at most cfg["gradient_tolerance"],
an L-BFGS round no longer changes the objective, or cfg["lbfgs_rounds"] rounds are done (recorded in run.json). No
early stopping: the validation NLL is recorded after each round and used only to choose c.
Writes <out>/scores.npz ((N, C), -inf padding), <out>/run.json, <out>/curve.jsonl.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
import math
import random
import time

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.utils import softmax as pyg_softmax

import fit_b0
from common import ROOT, load
from history_stats import KINDS
from model_data import load_meta, write_json

SLOT_TERMS = ["zone", "steps", "opp"]
PLAYER_TERMS = ["zone", "steps", "phase", "opp"]
CELLS = {"zone": 9, "steps": 5, "phase": 3, "opp": 5}
VARIANTS = {"m2": "types ids", "m2_main": "types ids", "m2_no_types": "ids", "m2_no_opp": "types ids",
            "m3": "types hist", "m23": "types ids hist", "m4": "types ids hist mp knn", "m4_no_knn": "types ids hist mp",
            "m4_no_id": "types hist mp knn", "d": "types ids"}
NODE_TYPES = ["player", "slot", "team", "role", "league"]
RELATIONS = {0: ("player", "in", "slot"), 1: ("slot", "of", "team"), 2: ("slot", "is", "role"),
             3: ("player", "plays", "role"), 4: ("team", "in", "league"), 5: ("player", "similar", "player")}
REVERSE = {0: ("slot", "holds", "player"), 1: ("team", "has", "slot"), 2: ("role", "of", "slot"),
           3: ("role", "played_by", "player"), 4: ("league", "has", "team"), 5: ("player", "similar", "player")}
SELF = {t: (t, "self", t) for t in NODE_TYPES}
METADATA_MP = (NODE_TYPES, sorted(set(RELATIONS.values()) | set(REVERSE.values()) | set(SELF.values())))
METADATA_SELF = (["player", "slot"], sorted([SELF["player"], SELF["slot"]]))


def deterministic(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class FastHGTConv(nn.Module):
    """PyG 2.8 HGTConv mathematics on its own parameters, one batched transform per relation
    (graph_prediction_research/five_task_hgt_v1/src/models.py), plus optional per-edge log-weights added to the
    attention scores (absent = 0, which is PyG's HGTConv; tests/test_model_math.py)."""

    def __init__(self, hidden, metadata, heads):
        super().__init__()
        self.conv = HGTConv(hidden, hidden, metadata, heads=heads)
        self.heads = heads
        self.dim = hidden // heads
        self.relation_index = {tuple(e): i for i, e in enumerate(metadata[1])}

    def forward(self, x_dict, edge_index_dict, edge_logw_dict=None):
        c, H, D = self.conv, self.heads, self.dim
        R = len(self.relation_index)
        k, q, v, offset = {}, {}, {}, {}
        total = 0
        for t, x in x_dict.items():
            lin = c.kqv_lin.lins[t]
            k[t], q[t], v[t] = F.linear(x, lin.weight, lin.bias).view(-1, 3, H, D).unbind(1)
            offset[t] = total
            total += x.shape[0]
        device = next(iter(x_dict.values())).device
        heads = torch.arange(H, device=device) * R
        by_source = {}
        for edge_type, index in edge_index_dict.items():
            by_source.setdefault(edge_type[0], []).append((edge_type, index))
        keys, values, targets, priors, biases = [], [], [], [], []
        for s, items in by_source.items():
            n, m = k[s].shape[0], len(items)
            chosen = (heads[:, None] + torch.tensor([self.relation_index[e] for e, _ in items], device=device)[None]).flatten()
            wk = c.k_rel.weight[chosen].view(H, m, D, D).permute(0, 2, 1, 3).reshape(H, D, m * D)
            wv = c.v_rel.weight[chosen].view(H, m, D, D).permute(0, 2, 1, 3).reshape(H, D, m * D)
            ks = torch.bmm(k[s].transpose(0, 1), wk).view(H, n, m, D).permute(1, 2, 0, 3).reshape(n * m, H, D)
            vs = torch.bmm(v[s].transpose(0, 1), wv).view(H, n, m, D).permute(1, 2, 0, 3).reshape(n * m, H, D)
            flat = torch.cat([index[0] * m + j for j, (_, index) in enumerate(items)])
            keys.append(ks[flat])
            values.append(vs[flat])
            targets.append(torch.cat([index[1] + offset[e[2]] for e, index in items]))
            priors.append(torch.cat([c.p_rel['__'.join(e)].expand(index.shape[1], H) for e, index in items]))
            biases.append(torch.cat([(edge_logw_dict[e] if edge_logw_dict is not None and e in edge_logw_dict
                                      else torch.zeros(index.shape[1], device=device)) for e, index in items]))
        result = {}
        if keys:
            target = torch.cat(targets)
            query = torch.cat([q[t] for t in x_dict])[target]
            score = (query * torch.cat(keys)).sum(-1) * torch.cat(priors) / math.sqrt(D) + torch.cat(biases)[:, None]
            message = torch.cat(values)
            weight = pyg_softmax(score, target, num_nodes=total)
            out = torch.zeros(total, H, D, device=score.device, dtype=score.dtype).index_add_(0, target, message * weight[..., None])
            out = out.view(total, H * D)
        else:
            out = torch.zeros(total, H * D, device=next(iter(x_dict.values())).device)
        for (t, x), o in zip(x_dict.items(), torch.split(out, [x.shape[0] for x in x_dict.values()])):
            if t not in c.dst_node_types:
                continue
            lin = c.out_lin.lins[t]
            alpha = c.skip[t].sigmoid()
            result[t] = alpha * F.linear(F.gelu(o), lin.weight, lin.bias) + (1 - alpha) * x
        return result


class Data:
    """Per target and per (target, slot) tensors on the device; the history graphs when a variant needs them."""

    def __init__(self, cfg, device, dtype, need_history, need_context):
        Y, meta = load_meta(cfg["scopes"])
        E = dict(np.load(ROOT / cfg["terms"]))
        off = np.load(ROOT / cfg["offsets"])
        O = off["scores"]
        self.mask_np = np.isfinite(O)
        if not np.array_equal(self.mask_np, E["player"] >= 0):
            raise ValueError("offsets and entity terms cover different candidate slots")
        t = lambda a, dt=torch.long: torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=dt)
        self.device = device
        self.offset = t(np.where(self.mask_np, O, 0.0), dtype)
        self.mask = t(self.mask_np, torch.bool)
        self.cell = {"zone": t(E["zone"]), "steps": t(E["steps"]), "phase": t(E["phase"]), "opp": t(np.maximum(E["opp5"], 0))}
        self.opp_on = t(E["opp5"] >= 0, torch.bool)
        for k in ("player", "slot", "role", "team_type"):
            setattr(self, k, t(np.maximum(E[k], 0)))
        self.phys = t(np.stack([E["left_foot"], E["height_z"], E["weight_z"], E["height_missing"], E["weight_missing"]], 1), dtype)
        self.sizes = {"player": len(E["player_ids"]), "slot": len(E["slot_ids"]), "team": len(E["team_ids"]),
                      "role": int(E["role"].max()) + 1, "team_type": int(E["team_type"].max()) + 1,
                      "league": int(E["league"].max()) + 1}
        self.actor = t(Y["actor_slot"])
        block = meta["block"].to_numpy()
        reach = Y["actor_slot"] >= 0
        self.train = np.flatnonzero((block == 0) & reach)
        self.val = np.flatnonzero((block == 1) & reach)
        self.N, self.C = O.shape
        self.E = E
        if need_context:
            self.context = t(off["context"], dtype)
        self.history = None
        if need_history:
            H = np.load(ROOT / cfg["history"])
            kind = H["node_kind"]
            self.history = {"feat": t(H["node_feat"], dtype), "id": t(H["node_id"]),
                            "player_row": t(H["player_row"]), "slot_row": t(H["slot_row"])}  # -1: no node, or padding
            local = np.zeros(len(kind), np.int64)  # position of each node among the nodes of its type
            self.type_rows = {}
            for name, code in KINDS.items():
                rows = np.flatnonzero(kind == code)
                local[rows] = np.arange(len(rows))
                self.type_rows[name] = t(rows)
            self.local = t(local)
            edges = H["edges"]
            if not (edges[:, 3] > 0).all():
                raise ValueError("history edges need positive weights")
            self.relations = {}
            for code, rel in RELATIONS.items():
                e = edges[edges[:, 0] == code]
                src, dst, lw = local[e[:, 1].astype(int)], local[e[:, 2].astype(int)], np.log(e[:, 3])
                if REVERSE[code] == rel:
                    self.relations[rel] = (t(np.stack([np.r_[src, dst], np.r_[dst, src]])), t(np.r_[lw, lw], dtype))
                else:
                    self.relations[rel] = (t(np.stack([src, dst])), t(lw, dtype))
                    self.relations[REVERSE[code]] = (t(np.stack([dst, src])), t(lw, dtype))


class EntityModel(nn.Module):
    def __init__(self, sizes, width, variant, n_feat):
        super().__init__()
        parts = set(VARIANTS[variant].split())
        self.variant, self.parts = variant, parts
        self.penalised_parameters = []
        if "ids" in parts:
            self.r_player = nn.Embedding(sizes["player"], width); self.r_slot = nn.Embedding(sizes["slot"], width)
            for e in (self.r_player, self.r_slot):
                nn.init.zeros_(e.weight)
            self.penalised_parameters += [self.r_player.weight, self.r_slot.weight]
        if "types" in parts:
            self.role = nn.Embedding(sizes["role"], width)
            self.style_role = nn.Embedding(sizes["team_type"] * sizes["role"], width)
            self.physique = nn.Linear(5, width, bias=False)
            for w in (self.role.weight, self.style_role.weight, self.physique.weight):
                nn.init.normal_(w, std=0.01)
            self.n_roles = sizes["role"]
            self.penalised_parameters += [self.role.weight, self.style_role.weight, self.physique.weight]
        if "hist" in parts:
            self.mp, self.knn = "mp" in parts, "knn" in parts
            self.types = NODE_TYPES if self.mp else ["player", "slot"]
            self.inp = nn.ModuleDict({"player": nn.Linear(n_feat, width), "slot": nn.Linear(n_feat, width)})
            if self.mp:
                self.node_emb = nn.ModuleDict({"team": nn.Embedding(sizes["team"], width), "role": nn.Embedding(sizes["role"], width),
                                               "league": nn.Embedding(sizes["league"], width)})
            metadata = METADATA_MP if self.mp else METADATA_SELF
            self.convs = nn.ModuleList([FastHGTConv(width, metadata, heads=2) for _ in range(2)])
            self.norms = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(width) for t in self.types}) for _ in range(2)])
        main = variant == "m2_main"
        no_opp = variant == "m2_no_opp"
        self.terms_s = ["zone"] if main else [t for t in SLOT_TERMS if not (no_opp and t == "opp")]
        self.terms_p = ["zone"] if main else [t for t in PLAYER_TERMS if not (no_opp and t == "opp")]
        self.A_s = nn.ParameterDict({t: nn.Parameter(torch.randn(CELLS[t], width) * 0.01) for t in self.terms_s})
        self.A_p = nn.ParameterDict({t: nn.Parameter(torch.randn(CELLS[t], width) * 0.01) for t in self.terms_p})
        if variant == "d":
            self.cell_emb = nn.ModuleDict({t: nn.Embedding(CELLS[t] + 1, 8) for t in CELLS})
            self.head = nn.Sequential(nn.Linear(64 + 2 * width + 8 * len(CELLS), 64), nn.GELU(), nn.Linear(64, 1))

    def penalised(self):
        """(level, vectors) of the score-space ridge; the physique map's columns are the effects of its five inputs."""
        out = []
        if "ids" in self.parts:
            out += [("player", self.r_player.weight), ("slot", self.r_slot.weight)]
        if "types" in self.parts:
            out += [("type", self.role.weight), ("type", self.style_role.weight), ("physique", self.physique.weight.T)]
        return out

    def shared(self):
        """Parameters outside the score-space ridge (A tables, encoder, the diagnostic head)."""
        skip = {id(p) for p in self.penalised_parameters}
        return [p for p in self.parameters() if id(p) not in skip]

    def ridge(self, lam):
        """1/2 sum over penalised vectors v and cells z of the terms t they affect of lam[(level, t)] <a_t[z], v>^2."""
        if self.variant == "d":
            return 0.5 * lam["d"] * sum((v ** 2).sum() for _, v in self.penalised())
        total = 0.0
        for level, v in self.penalised():
            G = 0.0
            for t in dict.fromkeys(self.terms_s + self.terms_p):
                if level in ("player", "physique"):
                    if t not in self.terms_p:
                        continue
                    a = self.A_p[t]
                else:
                    a = (self.A_s[t] if t in self.terms_s else 0.0) + (self.A_p[t] if t in self.terms_p else 0.0)
                G = G + lam[(level, t)] * (a.T @ a)
            total = total + ((v @ G) * v).sum()
        return 0.5 * total

    def encode(self, D):
        """Encoder outputs of the player and slot nodes of all snapshots, with a zero row in front for 'no node'."""
        x, edge_index, edge_logw = {}, {}, {}
        for t in self.types:
            r = D.type_rows[t]
            x[t] = self.inp[t](D.history["feat"][r]) if t in ("player", "slot") else self.node_emb[t](D.history["id"][r])
            loop = torch.arange(len(r), device=D.device)
            edge_index[SELF[t]] = torch.stack([loop, loop])
        if self.mp:
            for rel, (index, logw) in D.relations.items():
                if rel == RELATIONS[5] and not self.knn:
                    continue
                edge_index[rel], edge_logw[rel] = index, logw
        for conv, norm in zip(self.convs, self.norms):
            upd = conv(x, edge_index, edge_logw)
            x = {t: norm[t](x[t] + upd[t]) for t in x}
        return {t: torch.cat([torch.zeros(1, x[t].shape[1], device=D.device, dtype=x[t].dtype), x[t]]) for t in ("player", "slot")}

    def representations(self, D, idx, enc):
        player, slot, role, tt = (getattr(D, k)[idx] for k in ("player", "slot", "role", "team_type"))
        shape = player.shape + (self.A_s["zone"].shape[1],)
        e_p = torch.zeros(shape, device=D.device, dtype=D.offset.dtype)
        e_s = torch.zeros_like(e_p)
        if "ids" in self.parts:
            e_p = e_p + self.r_player(player)
            e_s = e_s + self.r_slot(slot)
        if "types" in self.parts:
            e_p = e_p + self.physique(D.phys[player])
            e_s = e_s + self.role(role) + self.style_role(tt * self.n_roles + role)
        if "hist" in self.parts:
            for t, key in (("player", "player_row"), ("slot", "slot_row")):
                r = D.history[key][idx]
                h = enc[t][torch.where(r >= 0, D.local[r.clamp(min=0)] + 1, 0)]
                if t == "player":
                    e_p = e_p + h
                else:
                    e_s = e_s + h
        return e_p + e_s, e_s  # the player inherits his slot's representation

    def forward(self, D, idx, enc=None):
        e_p, e_s = self.representations(D, idx, enc)
        if self.variant == "d":
            cells = [self.cell_emb[t](D.cell[t][idx] if t != "opp" else torch.where(D.opp_on[idx], D.cell[t][idx], CELLS[t]))
                     for t in CELLS]
            ctx = torch.cat([D.context[idx]] + cells, -1)
            C = e_p.shape[1]
            return self.head(torch.cat([ctx[:, None, :64].expand(-1, C, -1), e_p, e_s,
                                        ctx[:, None, 64:].expand(-1, C, -1)], -1)).squeeze(-1)
        out = torch.zeros(e_p.shape[:2], device=e_p.device, dtype=e_p.dtype)
        for terms, A, e in ((self.terms_s, self.A_s, e_s), (self.terms_p, self.A_p, e_p)):
            for t in terms:
                contrib = (e * A[t][D.cell[t][idx]][:, None, :]).sum(-1)
                out = out + (contrib * D.opp_on[idx][:, None] if t == "opp" else contrib)
        return out


def ridge_strengths(b0_lambdas, scale):
    """lam[(level, term)] of EntityModel.ridge: B0's cross-validated strengths times the common scale c."""
    lam = {"d": scale * b0_lambdas["player_zone"]}
    for t in CELLS:
        lam[("player", t)] = scale * b0_lambdas[f"player_{t}"]
        lam[("slot", t)] = scale * b0_lambdas[f"slot_{t}" if t in SLOT_TERMS else f"player_{t}"]
        lam[("physique", t)] = scale * b0_lambdas["physique_zone"]
        lam[("type", t)] = scale * b0_lambdas["type_steps"]
    return lam


def nll(D, idx, entity):
    scores = (D.offset[idx] + entity).masked_fill(~D.mask[idx], float("-inf"))
    return -F.log_softmax(scores, -1).gather(1, D.actor[idx][:, None]).squeeze(1)


def chunks(rows, size):
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def objective(model, D, lam, cfg, backward):
    """Mean NLL of the reachable training targets + ridge / N + weight decay of the shared parameters. With backward,
    the gradients are accumulated over chunks of targets (the encoder output is computed once per evaluation)."""
    N = len(D.train)
    enc = model.encode(D) if "hist" in model.parts else None
    parts = chunks(D.train, cfg["chunk_targets"])
    total = 0.0
    for j, b in enumerate(parts):
        idx = torch.as_tensor(b, device=D.device)
        loss = nll(D, idx, model(D, idx, enc)).sum() / N
        if backward:
            loss.backward(retain_graph=enc is not None and j < len(parts) - 1)
        total += float(loss)
    reg = model.ridge(lam) / N + 0.5 * cfg["weight_decay"] * sum((p ** 2).sum() for p in model.shared())
    if backward:
        reg.backward()
    return total + float(reg), total


@torch.no_grad()
def evaluate(model, D, rows, cfg):
    enc = model.encode(D) if "hist" in model.parts else None
    total = 0.0
    for b in chunks(rows, cfg["chunk_targets"]):
        idx = torch.as_tensor(b, device=D.device)
        total += float(nll(D, idx, model(D, idx, enc)).sum())
    return total / len(rows)


@torch.no_grad()
def all_scores(model, D, cfg):
    enc = model.encode(D) if "hist" in model.parts else None
    out = np.full((D.N, D.C), -np.inf, np.float32)
    for b in chunks(np.arange(D.N), cfg["chunk_targets"]):
        idx = torch.as_tensor(b, device=D.device)
        out[b] = (D.offset[idx] + model(D, idx, enc)).masked_fill(~D.mask[idx], float("-inf")).float().cpu().numpy()
    return out


def distill(model, D, cfg, rng, curve):
    """Least squares of the entity scores against B0's entity part, both centred within each target."""
    b0 = torch.as_tensor(np.load(ROOT / cfg["b0"])["entity_part"], device=D.device, dtype=D.offset.dtype)
    m = D.mask.to(b0.dtype)
    b0 = b0 - (b0 * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["distill_lr"])
    for epoch in range(1, cfg["distill_epochs"] + 1):
        total, n = 0.0, 0
        for b in chunks(rng.permutation(D.train), cfg["distill_batch"]):
            idx = torch.as_tensor(b, device=D.device)
            enc = model.encode(D) if "hist" in model.parts else None
            mi = m[idx]
            ent = model(D, idx, enc) * mi
            ent = ent - ent.sum(1, keepdim=True) / mi.sum(1, keepdim=True)
            loss = (((ent - b0[idx]) * mi) ** 2).sum() / mi.sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += float(loss) * len(b); n += len(b)
        rec = {"stage": "distill", "epoch": epoch, "mse": total / n}
        curve.write(json.dumps(rec) + "\n"); curve.flush(); print(json.dumps(rec), flush=True)
    return total / n


def train(cfg, args, D, out):
    model = EntityModel(D.sizes, cfg["width"], args.model, 44).to(D.device)
    lam = ridge_strengths(load(ROOT / cfg["b0_json"])["lambdas"], args.ridge_scale)
    rng = np.random.default_rng(args.seed)
    curve = open(out / "curve.jsonl", "w", encoding="utf-8")
    mse = distill(model, D, cfg, rng, curve)
    rec = {"stage": "distilled", "validation_nll": evaluate(model, D, D.val, cfg)}
    curve.write(json.dumps(rec) + "\n"); curve.flush(); print(json.dumps(rec), flush=True)
    params = list(model.parameters())
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg["lbfgs_round_iterations"], tolerance_grad=cfg["gradient_tolerance"],
                            tolerance_change=1e-12, history_size=cfg["lbfgs_history"], line_search_fn="strong_wolfe")
    calls = [0]

    def closure():
        opt.zero_grad(set_to_none=False)
        calls[0] += 1
        return torch.tensor(objective(model, D, lam, cfg, backward=True)[0])

    stop, previous = "round limit", None
    for rnd in range(1, cfg["lbfgs_rounds"] + 1):
        opt.step(closure)
        opt.zero_grad(set_to_none=False)
        f, train_nll = objective(model, D, lam, cfg, backward=True)
        g = max(float(p.grad.abs().max()) for p in params if p.grad is not None)
        rec = {"stage": "lbfgs", "round": rnd, "function_calls": calls[0], "objective": f, "train_nll": train_nll,
               "gradient_max": g, "validation_nll": evaluate(model, D, D.val, cfg)}
        curve.write(json.dumps(rec) + "\n"); curve.flush(); print(json.dumps(rec), flush=True)
        if g <= cfg["gradient_tolerance"]:
            stop = "gradient tolerance"
            break
        if previous is not None and f >= previous:
            stop = "no change in a round"
            break
        previous = f
    curve.close()
    np.savez(out / "scores.npz", scores=all_scores(model, D, cfg))
    return {"validation_nll": rec["validation_nll"], "objective": rec["objective"], "train_nll": rec["train_nll"],
            "gradient_max": rec["gradient_max"], "stopped_by": stop, "rounds": rnd, "function_calls": calls[0],
            "distill_mse": mse, "ridge_strengths": {str(k): v for k, v in lam.items()},
            "parameters": int(sum(p.numel() for p in params))}


def degenerate(cfg, args, D, out):
    """B0 in PyTorch (float64, full batch, L-BFGS) with B0's terms and ridge strengths; compared with fit_b0.py."""
    info_b0 = load(ROOT / cfg["b0_json"])
    theta_ref = np.load(ROOT / cfg["b0_theta"])["theta"]
    rows_t, slots = np.nonzero(D.mask_np)
    B = fit_b0.blocks(D.E, rows_t, slots)
    X = sparse.hstack([B[t] for t in fit_b0.TERMS]).tocsr()
    lam = np.concatenate([np.full(B[t].shape[1], info_b0["lambdas"][t]) for t in fit_b0.TERMS])
    fit_rows = np.isin(rows_t, D.train)
    Xf = X[fit_rows].tocoo()
    dev = D.device
    Xt = torch.sparse_coo_tensor(np.vstack([Xf.row, Xf.col]), Xf.data, Xf.shape, dtype=torch.float64, device=dev).coalesce()
    off = torch.as_tensor(np.load(ROOT / cfg["offsets"])["scores"][rows_t[fit_rows], slots[fit_rows]], dtype=torch.float64, device=dev)
    group = torch.as_tensor(np.searchsorted(np.unique(rows_t[fit_rows]), rows_t[fit_rows]), device=dev)
    actor = D.actor.cpu().numpy()
    y = torch.as_tensor((slots[fit_rows] == actor[rows_t[fit_rows]]) * 1.0, dtype=torch.float64, device=dev)
    G = int(group.max()) + 1
    lam_t = torch.as_tensor(lam, dtype=torch.float64, device=dev)
    theta = torch.zeros(X.shape[1], dtype=torch.float64, device=dev, requires_grad=True)

    def objective():
        s = off + torch.sparse.mm(Xt, theta[:, None]).squeeze(1)
        m = torch.full((G,), -torch.inf, dtype=torch.float64, device=dev).scatter_reduce(0, group, s, "amax")
        z = torch.zeros(G, dtype=torch.float64, device=dev).index_add(0, group, torch.exp(s - m[group]))
        return (m + torch.log(z)).sum() - (s * y).sum() + 0.5 * (lam_t * theta * theta).sum()

    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=cfg["lbfgs_iterations"], tolerance_grad=1e-12, tolerance_change=1e-15,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); f = objective(); f.backward(); return f
    opt.step(closure)
    f = float(objective())
    th = theta.detach().cpu().numpy()
    with torch.no_grad():
        theta.copy_(torch.as_tensor(theta_ref, dtype=torch.float64, device=dev))
        f_ref = float(objective())
    result = {"max_parameter_difference": float(np.abs(th - theta_ref).max()),
              "loglik_difference_per_action": abs(f - f_ref) / G, "torch_objective": f, "scipy_objective": f_ref,
              "targets": int(G), "parameters": int(len(th)),
              "passed": bool(np.abs(th - theta_ref).max() < 1e-4 and abs(f - f_ref) / G < 1e-6)}
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=list(VARIANTS) + ["degenerate"])
    parser.add_argument("--ridge-scale", type=float, help="c, the common factor on B0's ridge strengths")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.model != "degenerate" and args.ridge_scale is None:
        parser.error("--ridge-scale is required for the entity variants")
    cfg = load(ROOT / args.config)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    deterministic(args.seed)
    if args.model == "degenerate":
        torch.use_deterministic_algorithms(False)  # a convex solve compared within a tolerance; scatter ops allowed
    device = torch.device(args.device)
    parts = set(VARIANTS.get(args.model, "").split())
    D = Data(cfg, device, torch.float64 if args.model == "degenerate" else torch.float32,
             need_history="hist" in parts, need_context=args.model == "d")
    t0 = time.time()
    result = degenerate(cfg, args, D, out) if args.model == "degenerate" else train(cfg, args, D, out)
    result.update({"model": args.model, "ridge_scale": args.ridge_scale, "seed": args.seed, "config": args.config,
                   "wall_seconds": round(time.time() - t0, 1)})
    write_json(out / "run.json", result)
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
