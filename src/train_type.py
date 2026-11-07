"""Train an action-type model of src/type_model.py on the whole training period (docs/MODEL_PLAN.md, note 22).

Same sampler, schedule, optimiser and determinism as src/train_seq.py. Loss: the type NLL (for the conditional head,
of the true actor's conditional) plus, when cfg["who_loss"], the who NLL. Labels: the action class of
targets_label.parquet (src/event_schema.py ACTIONS), read only to form the loss and the evaluation. The checkpoint with
the lowest validation type NLL (the mixture for the conditional head) is kept. Writes to --out: run.json, curve.jsonl,
best.pt, predictions.npz (validation rows; log P(type) (N, T); for the conditional head also the true-actor
conditional; who scores).
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from common import ROOT, load
from event_schema import ACTIONS
from model_data import load_scopes, scope_dirs, write_json
from seq_model import actor_loss
from train_seq import deterministic, gather, lr_at, rotate_orders
from type_model import WhoType, type_log_probs


def type_labels(scope_configs, actor_slot):
    labels = pd.concat([pd.read_parquet(d / "targets_label.parquet", columns=["action", "actor_slot"]) for d in scope_dirs(scope_configs)],
                       ignore_index=True)
    if not np.array_equal(labels["actor_slot"].to_numpy(), actor_slot):
        raise ValueError("targets_label and arrays_label are not aligned")
    return labels["action"].map({a: i for i, a in enumerate(ACTIONS)}).to_numpy().astype(np.int64)


@torch.no_grad()
def evaluate(model, T, rows, actor, label, device, batch=2048):
    model.eval()
    lp, oracle, who = [], [], []
    for s in range(0, len(rows), batch):
        idx = torch.as_tensor(rows[s:s + batch], device=device)
        w, logits = model(gather(T, idx))
        lp.append(type_log_probs(model, w, logits).cpu())
        if model.head == "conditional":
            oracle.append(type_log_probs(model, w, logits, actor[idx]).cpu())
        who.append(w.cpu())
    model.train()
    lp = torch.cat(lp)
    nll = float(-lp[torch.arange(len(rows)), torch.as_tensor(label[rows])].mean())
    return nll, lp.numpy(), (torch.cat(oracle).numpy() if oracle else None), torch.cat(who).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    if cfg["type_head"] == "conditional" and not cfg["who_loss"]:
        raise ValueError("the conditional head needs the who loss (its mixture uses the who-head)")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    deterministic(args.seed)
    device = torch.device(args.device)

    X, Y, meta, shared = load_scopes(cfg["scopes"])
    actor = Y["actor_slot"]
    label = type_labels(cfg["scopes"], actor)
    block = meta["block"].to_numpy()
    reach = actor >= 0
    train_rows = np.flatnonzero((block == 0) & reach)
    val_rows = np.flatnonzero((block == 1) & reach)
    u_mean, u_sd = float(X["urgency"][block == 0].mean()), float(X["urgency"][block == 0].std())
    X["urgency"] = ((X["urgency"] - u_mean) / u_sd).astype(np.float32)
    T = {k: torch.as_tensor(v, device=device) for k, v in X.items()}
    actor_t = torch.as_tensor(actor, device=device)
    label_t = torch.as_tensor(label, device=device)

    model = WhoType(cfg["model"], shared, cfg["type_head"], len(ACTIONS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    orders = train_rows[rotate_orders(meta["match_id"].to_numpy()[train_rows], args.seed, cfg["epochs"], cfg["targets_per_match"])]
    run = {"config": args.config, "seed": args.seed, "type_head": cfg["type_head"], "who_loss": cfg["who_loss"],
           "train_targets": int(len(train_rows)), "validation_targets": int(len(val_rows)),
           "parameters": int(sum(p.numel() for p in model.parameters()))}
    write_json(out / "run.json", run)
    best, best_epoch = np.inf, 0
    curve = open(out / "curve.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        lr = lr_at(cfg, epoch)
        for group in opt.param_groups:
            group["lr"] = lr
        order = orders[epoch - 1]
        tot_type, tot_who, n = 0.0, 0.0, 0
        for s in range(0, len(order) - cfg["batch"] + 1, cfg["batch"]):
            idx = torch.as_tensor(order[s:s + cfg["batch"]], device=device)
            who, logits = model(gather(T, idx))
            if model.head == "conditional":
                logits = logits[torch.arange(len(idx), device=device), actor_t[idx]]
            loss_type = F.cross_entropy(logits, label_t[idx])
            loss_who = actor_loss(who, actor_t[idx]).mean()
            loss = loss_type + (loss_who if cfg["who_loss"] else 0.0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"], error_if_nonfinite=True)
            opt.step()
            tot_type += float(loss_type) * len(idx); tot_who += float(loss_who) * len(idx); n += len(idx)
        record = {"epoch": epoch, "lr": lr, "train_type_nll": tot_type / n, "train_who_nll": tot_who / n,
                  "seconds": round(time.time() - t0, 1)}
        if epoch % cfg["validate_every"] == 0:
            v, _, _, w = evaluate(model, T, val_rows, actor_t, label, device)
            record["validation_type_nll"] = v
            record["validation_who_nll"] = float(actor_loss(torch.as_tensor(w), torch.as_tensor(actor[val_rows])).mean())
            if v < best:
                best, best_epoch = v, epoch
                torch.save(model.state_dict(), out / "best.pt")
        curve.write(json.dumps(record) + "\n"); curve.flush()
        print(json.dumps(record), flush=True)
    curve.close()
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    v, lp, oracle, who = evaluate(model, T, val_rows, actor_t, label, device)
    saved = {"rows": val_rows, "type_log_probs": lp.astype(np.float32), "who": who.astype(np.float32)}
    if oracle is not None:
        saved["type_log_probs_true_actor"] = oracle.astype(np.float32)
    np.savez(out / "predictions.npz", **saved)
    run.update({"best_epoch": best_epoch, "best_validation_type_nll": best, "wall_seconds": round(time.time() - t0, 1)})
    write_json(out / "run.json", run)
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
