"""Train M1 (src/seq_model.py) for one cross-fitting fold or on the whole training period (docs/MODEL_PLAN.md, section 4).

--fold k     trains on the training matches outside fold k and predicts every target of fold k (out-of-fold offsets)
--fold full  trains on all training matches and predicts every validation target
Training: AdamW (weight decay from the config), gradient clipping, the fixed step schedule, the Rotate sampler of
graph_prediction_research (128 targets per match and epoch, batches of 256), deterministic FP32. The checkpoint with the
lowest validation NLL (reachable validation targets, scheduled validations only) is kept; ties keep the earlier epoch.
Writes to --out: run.json, curve.jsonl, best.pt, predictions.npz (global target index, scores per slot, context vector).
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import hashlib
import json
import random
import time

import numpy as np
import torch

from common import ROOT, load
from model_data import folds, load_scopes, write_json
from seq_model import M1, actor_loss


def deterministic(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.backends.mkldnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def stable_seed(*parts):
    # Sampling seed only, as in graph_prediction_research/src/train.py; not artifact hashing.
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little")


def rotate_orders(match_id, seed, epochs, count):
    """graph_prediction_research/src/train.py rotate_orders with the per-match count as a parameter."""
    tables = []
    for mid in np.unique(match_id):
        rows = np.flatnonzero(match_id == mid); n = len(rows)
        if n < count:
            raise ValueError(f"match {mid} has fewer than {count} training targets")
        first_generator = torch.Generator().manual_seed(stable_seed(seed, "train", mid))
        slots = []
        for slot in range(count):
            lo, hi = slot * n // count, (slot + 1) * n // count
            first = lo + int(torch.randint(hi - lo, (1,), generator=first_generator))
            rest = [x for x in range(lo, hi) if x != first]
            order = [first, *np.random.default_rng(stable_seed(seed, mid, slot)).permutation(rest).tolist()]
            slots.append(rows[np.array([order[e % len(order)] for e in range(epochs)])])
        tables.append(np.stack(slots, 1))
    generator = torch.Generator().manual_seed(seed); result = []
    for e in range(epochs):
        result.append(np.concatenate([tables[m][e][torch.randperm(count, generator=generator).numpy()]
                                      for m in torch.randperm(len(tables), generator=generator).tolist()]))
    return np.stack(result)


def lr_at(cfg, epoch):
    return cfg["lr"] * cfg["factor"] ** sum(epoch > m for m in cfg["milestones"])


def gather(T, idx):
    return {k: v[idx] for k, v in T.items()}


@torch.no_grad()
def predict(model, T, rows, device, batch=2048):
    model.eval()
    scores, context = [], []
    for s in range(0, len(rows), batch):
        idx = torch.as_tensor(rows[s:s + batch], device=device)
        sc, c = model(gather(T, idx))
        scores.append(sc.cpu()); context.append(c.cpu())
    model.train()
    return torch.cat(scores).numpy(), torch.cat(context).numpy()


def mean_nll(scores, actor_slot):
    s = torch.as_tensor(scores)
    return float(actor_loss(s, torch.as_tensor(actor_slot)).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    deterministic(args.seed)
    device = torch.device(args.device)

    X, Y, meta, shared = load_scopes(cfg["scopes"])
    fold = folds(meta, cfg["folds"], cfg["fold_seed"])
    block = meta["block"].to_numpy()
    actor = Y["actor_slot"]
    reach = actor >= 0
    if args.fold == "full":
        train_rows = np.flatnonzero((block == 0) & reach)
        predict_rows = np.flatnonzero(block == 1)
    else:
        f = int(args.fold)
        train_rows = np.flatnonzero((block == 0) & reach & (fold != f))
        predict_rows = np.flatnonzero((block == 0) & (fold == f))
    val_rows = np.flatnonzero((block == 1) & reach)
    u_mean, u_sd = float(X["urgency"][block == 0].mean()), float(X["urgency"][block == 0].std())
    X["urgency"] = ((X["urgency"] - u_mean) / u_sd).astype(np.float32)
    T = {k: torch.as_tensor(v, device=device) for k, v in X.items()}
    actor_t = torch.as_tensor(actor, device=device)

    model = M1(cfg["model"], shared).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    orders = rotate_orders(meta["match_id"].to_numpy()[train_rows], args.seed, cfg["epochs"], cfg["targets_per_match"])
    orders = train_rows[orders]
    run = {"config": args.config, "fold": args.fold, "seed": args.seed, "train_targets": int(len(train_rows)),
           "train_matches": int(len(np.unique(meta["match_id"].to_numpy()[train_rows]))),
           "predict_targets": int(len(predict_rows)), "validation_targets": int(len(val_rows)),
           "urgency_mean": u_mean, "urgency_sd": u_sd, "parameters": int(sum(p.numel() for p in model.parameters())),
           "steps_per_epoch": int(orders.shape[1] // cfg["batch"])}
    write_json(out / "run.json", run)
    best, best_epoch = np.inf, 0
    curve = open(out / "curve.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        lr = lr_at(cfg, epoch)
        for group in opt.param_groups:
            group["lr"] = lr
        order = orders[epoch - 1]
        total, n = 0.0, 0
        for s in range(0, len(order) - cfg["batch"] + 1, cfg["batch"]):
            idx = torch.as_tensor(order[s:s + cfg["batch"]], device=device)
            scores, _ = model(gather(T, idx))
            loss = actor_loss(scores, actor_t[idx]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"], error_if_nonfinite=True)
            opt.step()
            total += float(loss) * len(idx); n += len(idx)
        record = {"epoch": epoch, "lr": lr, "train_nll": total / n, "seconds": round(time.time() - t0, 1)}
        if epoch in (1, 2) or epoch % cfg["validate_every"] == 0:
            vs, _ = predict(model, T, val_rows, device)
            record["validation_nll"] = mean_nll(vs, actor[val_rows])
            if epoch % cfg["validate_every"] == 0 and record["validation_nll"] < best:
                best, best_epoch = record["validation_nll"], epoch
                torch.save(model.state_dict(), out / "best.pt")
        curve.write(json.dumps(record) + "\n"); curve.flush()
        print(json.dumps(record), flush=True)
    curve.close()
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    scores, context = predict(model, T, predict_rows, device)
    np.savez(out / "predictions.npz", rows=predict_rows, scores=scores.astype(np.float32), context=context.astype(np.float32))
    run.update({"best_epoch": best_epoch, "best_validation_nll": best, "wall_seconds": round(time.time() - t0, 1)})
    write_json(out / "run.json", run)
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
