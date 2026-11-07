"""Frozen-checkpoint inference on explicit non-test scopes, with no target-league fitting."""
import argparse
import numpy as np
import torch

from actor_baselines import ActorBaseline
from actor_benchmark_data import prediction_archive, read_data, transform
from actor_benchmark_metrics import score_archive
from common import ROOT, load
from train_actor_baseline import predict, write_json
from train_seq import deterministic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', required=True, help='explicit target-scope input configuration')
    p.add_argument('--device', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    device = torch.device(args.device)
    saved = torch.load(ROOT / args.checkpoint, map_location='cpu', weights_only=True)
    if saved['mode'] != 'train':
        raise ValueError('a smoke checkpoint cannot be used for research inference')
    cfg = load(ROOT / args.config)
    if cfg['encoders'] != saved['config']['encoders'] or cfg['task'] != saved['config']['task']:
        raise ValueError('target scope uses a different native input schema')
    deterministic(saved['seed'])
    torch.backends.mha.set_fastpath_enabled(False)
    X, actor, meta = read_data(cfg, saved['family'], False)
    rows = np.flatnonzero(meta.block.to_numpy() == 1)
    pre = saved['preprocessing']
    if not len(rows) or meta.kickoff.iloc[rows].min() <= pre['train_last_kickoff']:
        raise ValueError('all scored matches must follow the source training cutoff')
    model = ActorBaseline(saved['family'], saved['config'], len(pre['player_ids'])).to(device)
    model.load_state_dict(saved['state_dict'])
    scores = predict(model, transform(X, pre), rows, device, saved['config']['prediction_batch'])
    archive = prediction_archive(X, actor, meta, rows, scores, pre)
    metrics, table = score_archive(archive)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / 'predictions.npz', **archive)
    write_json(out / 'metrics.json', metrics)
    table.to_csv(out / 'per_match.csv', index=False)
    write_json(out / 'inference.json', {'checkpoint': args.checkpoint, 'config': args.config,
                                       'training_player_count': len(pre['player_ids']),
                                       'target_league_fitting': False})


if __name__ == '__main__':
    main()
