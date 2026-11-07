"""Train or smoke-test one native-input conditional-actor adaptation; no implicit job launch.

Checkpoint selection uses reachable validation NLL. All validation targets remain in
predictions, including unreachable actors. Shared Rotate orders, schedules and LR
grids are defined by the experiment config. Inputs stay on CPU and only minibatches
move to the explicitly chosen device. No test scoring or automatic resumption.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
import time

import numpy as np
import torch

from actor_baselines import ActorBaseline, FAMILIES
from actor_benchmark_data import batch, fit_preprocessing, prediction_archive, read_data, transform
from actor_benchmark_metrics import score_archive
from common import ROOT, load
from seq_model import actor_loss
from train_seq import deterministic, rotate_orders


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


@torch.no_grad()
def predict(model, inputs, rows, device, batch_size):
    model.eval()
    result = []
    for start in range(0, len(rows), batch_size):
        score, _ = model(batch(inputs, rows[start:start + batch_size], device))
        result.append(score.cpu().numpy())
    return np.concatenate(result)


def choose_rows(meta, actor, cfg, mode, seed):
    train = np.flatnonzero((meta.block.to_numpy() == 0) & (actor >= 0))
    val = np.flatnonzero(meta.block.to_numpy() == 1)
    if mode == "smoke":
        rng = np.random.default_rng(seed)
        matches = np.sort(meta.match_id.iloc[train].unique())
        if len(matches) < cfg["smoke"]["train_matches"]:
            raise ValueError("not enough training matches for the explicit smoke setting")
        chosen = rng.choice(matches, cfg["smoke"]["train_matches"], replace=False)
        train = train[np.isin(meta.match_id.to_numpy()[train], chosen)]
        if len(val) < cfg["smoke"]["validation_targets"]:
            raise ValueError("not enough validation targets for the explicit smoke setting")
        val = np.sort(rng.choice(val, cfg["smoke"]["validation_targets"], replace=False))
    if not len(train) or not len(val) or not (actor[val] >= 0).any():
        raise ValueError("empty training or reachable validation set")
    return train, val


def train(cfg, family, seed, lr, mode, device, out, data=None):
    if cfg["task"] != "conditional_actor_A":
        raise ValueError("this runner accepts only the conditional-A view, not next-event forecasting")
    if family not in FAMILIES or lr not in cfg["learning_rates"]:
        raise ValueError("family or learning rate outside the explicit protocol")
    if mode == "train" and seed not in [cfg["search_seed"], *cfg["confirmation_seeds"]]:
        raise ValueError("training seed outside the explicit protocol")
    deterministic(seed)
    # Prevent PyTorch's inference-only attention fastpath from changing padding/batch behaviour.
    torch.backends.mha.set_fastpath_enabled(False)
    X, actor, meta = read_data(cfg, family, True) if data is None else data
    train_rows, val_rows = choose_rows(meta, actor, cfg, mode, seed)
    setting = cfg["training"] if mode == "train" else cfg["smoke"]
    preprocessing = fit_preprocessing(X, train_rows, meta.kickoff.iloc[train_rows].max(), family)
    inputs = transform(X, preprocessing)
    model = ActorBaseline(family, cfg, len(preprocessing["player_ids"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])
    orders = train_rows[rotate_orders(meta.match_id.to_numpy()[train_rows], seed, setting["epochs"], setting["targets_per_match"])]
    batch_size = setting["batch"]
    if orders.shape[1] % batch_size:
        raise ValueError("targets_per_match * training matches must divide batch size; no silent dropping")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    write_json(out / "preprocessing.json", preprocessing)
    np.savez(out / "training_rows.npz", match_id=meta.match_id.to_numpy()[train_rows], seq=meta.seq.to_numpy()[train_rows],
             event_id=meta.event_id.to_numpy()[train_rows])
    best, best_epoch, start_time = np.inf, 0, time.time()
    parameters = int(sum(p.numel() for p in model.parameters()))
    initial = predict(model, inputs, val_rows, device, cfg["prediction_batch"])
    val_actor = actor[val_rows]
    reachable = val_actor >= 0
    def validation_nll(scores):
        return float(actor_loss(torch.as_tensor(scores[reachable], dtype=torch.float64),
                                torch.as_tensor(val_actor[reachable])).mean())
    initial_nll = validation_nll(initial)
    with (out / "curve.jsonl").open("w", encoding="utf-8") as curve:
        for epoch in range(1, setting["epochs"] + 1):
            model.train()
            current_lr = lr * cfg["training"]["factor"] ** sum(epoch > m for m in setting["milestones"])
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            total = 0.0
            for offset in range(0, orders.shape[1], batch_size):
                rows = orders[epoch - 1, offset:offset + batch_size]
                score, _ = model(batch(inputs, rows, device))
                loss = actor_loss(score, torch.as_tensor(actor[rows], device=device)).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["clip"], error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * len(rows)
            record = {"epoch": epoch, "lr": current_lr, "training_nll": total / orders.shape[1],
                      "seconds": time.time() - start_time}
            if epoch % setting["validate_every"] == 0 or epoch == setting["epochs"]:
                scores = predict(model, inputs, val_rows, device, cfg["prediction_batch"])
                v = validation_nll(scores)
                if not np.isfinite(v):
                    raise RuntimeError("non-finite validation loss")
                record["validation_nll"] = v
                if v < best:
                    best, best_epoch = v, epoch
                    torch.save({"state_dict": model.state_dict(), "config": cfg,
                                "preprocessing": preprocessing, "family": family,
                                "seed": seed, "lr": lr, "epoch": epoch, "mode": mode}, out / "best.pt")
            curve.write(json.dumps(record) + "\n"); curve.flush()
            print(json.dumps({"family": family, **record}), flush=True)
    saved = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    scores = predict(model, inputs, val_rows, device, cfg["prediction_batch"])
    archive = prediction_archive(X, actor, meta, val_rows, scores, preprocessing)
    metrics, contributions = score_archive(archive)
    np.savez(out / "predictions.npz", **archive)
    write_json(out / "metrics.json", metrics)
    contributions.to_csv(out / "per_match.csv", index=False)
    result = {"task": cfg["task"], "family": family, "mode": mode, "seed": seed, "lr": lr,
              "parameters": parameters, "training_matches": int(meta.match_id.iloc[train_rows].nunique()),
              "training_targets": int(len(train_rows)), "validation_targets": int(len(val_rows)),
              "reachable_validation_targets": int(reachable.sum()), "initial_validation_nll": initial_nll,
              "best_validation_nll": best, "best_epoch": best_epoch, "epochs": setting["epochs"],
              "optimizer_steps": setting["epochs"] * orders.shape[1] // batch_size,
              "wall_seconds": time.time() - start_time,
              "interpretation": "software verification only" if mode == "smoke" else "development validation; not test performance"}
    write_json(out / "run.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--mode", choices=["smoke", "train"], required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(train(load(ROOT / a.config), a.family, a.seed, a.lr, a.mode, torch.device(a.device), ROOT / a.out))


if __name__ == "__main__":
    main()
