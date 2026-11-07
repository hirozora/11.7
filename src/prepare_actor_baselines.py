"""Build native-input caches from explicit non-test scopes; no fitting or training.

One directory per scope, with separate native feature files and evaluation metadata.
Only matches in the supplied model train/validation manifests are materialised.
"""
import argparse
from collections import defaultdict
import json

import numpy as np
import pandas as pd

from actor_native_inputs import NativeState, match_views
from build_actions import validate_card_overrides
from common import ROOT, load
from event_schema import PERIOD_ORDER


def prepare(cfg, dataset):
    spec = cfg['datasets'][dataset]
    scope = load(ROOT / spec['scope'])
    dc = load(ROOT / scope['data_config'])
    source, out = ROOT / scope['out_dir'], ROOT / spec['native_dir']
    if out.exists():
        raise FileExistsError(out)
    for filename, expected in [('checks.json', 11), ('checks_inputs.json', 4)]:
        checks = load(source / filename)
        if len(checks) != expected or not all(r['passed'] for r in checks.values()):
            raise ValueError(f'{source}: model-view checks have not passed')
    meta = pd.read_parquet(source / 'targets_meta.parquet')
    matches = pd.read_parquet(source / 'matches.parquet')
    test = {r['match_id'] for r in load(ROOT / scope['test_manifest'])}
    if set(matches.match_id) & test or set(meta.match_id) != set(matches.match_id):
        raise ValueError('invalid development match set')
    for block, key in [('model_train', 'train_manifest'), ('model_validation', 'validation_manifest')]:
        expected = {r['match_id']: r['kickoff'] for r in load(ROOT / scope[key])}
        if matches.loc[matches.block == block].set_index('match_id').kickoff.to_dict() != expected:
            raise ValueError(f'match table differs from {key}')
    ids = set(matches.match_id)
    events = defaultdict(list)
    positions = {r['event_id']: dict(x=r['x'], y=r['y']) for r in dc['position_overrides']}
    for e in load(ROOT / dc['raw_dir'] / 'events' / f"events_{dc['league']}.json"):
        if e['matchId'] in ids:
            if e['id'] in positions:
                e['positions'][0] = positions[e['id']]
            events[e['matchId']].append(e)
    for stream in events.values():
        stream.sort(key=lambda e: (PERIOD_ORDER[e['matchPeriod']], e['eventSec']))
    cards = validate_card_overrides(events, dc['card_overrides'])
    for stream in events.values():
        for e in stream:
            if e['id'] in cards:
                # The configured dismissal is explicit; retain any observed yellow tag.
                e['tags'] = [*e['tags'], {'id': 1701}]
    match_raw = {m['wyId']: m for m in load(ROOT / dc['raw_dir'] / 'matches' / f"matches_{dc['league']}.json")
                 if m['wyId'] in ids}
    seq_of = {e['id']: s for stream in events.values() for s, e in enumerate(stream)}
    row_of = dict(zip(meta.event_id, meta.index))
    exclusions = defaultdict(set)
    coinciding = pd.read_parquet(ROOT / dc['coinciding_path'])
    for target, record in zip(coinciding.target_event_id, coinciding.record_event_id):
        exclusions[row_of[target]].add(seq_of[record])
    windows = {k: v['window'] for k, v in cfg['encoders'].items()}
    empty = NativeState(1, 2, {}).view(1, windows)
    out.mkdir(parents=True, exist_ok=False)
    arrays = {family: {k: np.lib.format.open_memmap(out / f'{family}_{k}.npy', mode='w+',
                         dtype=v.dtype, shape=(len(meta), *v.shape)) for k, v in values.items()}
              for family, values in empty.items()}
    overrides = {r['event_id']: r['scorer_team'] for r in dc['goal_overrides']}
    for number, (mid, group) in enumerate(meta.groupby('match_id', sort=True), 1):
        sides = {t['side']: int(tid) for tid, t in match_raw[mid]['teamsData'].items()}
        stream = events[mid]
        for r in group.itertuples():
            if stream[r.seq]['id'] != r.event_id:
                raise ValueError('target key mismatch with raw stream')
        targets = list(zip(group.index, group.seq, group.team_id))
        for row, view in match_views(stream, targets, exclusions, sides['home'], sides['away'], overrides, windows):
            for family, values in view.items():
                for k, v in values.items():
                    arrays[family][k][row] = v
        if number % 20 == 0 or number == len(matches):
            print(json.dumps({'prepared_matches': number, 'total_matches': len(matches)}), flush=True)
    for values in arrays.values():
        for v in values.values():
            v.flush()
    # Only identity/mask and predeclared query conditions may enter the output head.
    with np.load(source / 'arrays_in.npz') as a:
        raw, mask = a['cand_player'], a['cand_mask']
    cand = dict(np.load(source / 'candidates_in.npz'))
    prior = np.zeros_like(raw, dtype=np.int32)
    recent = np.zeros_like(mask)
    prior[cand['act'], cand['slot']] = cand['prior_matches']
    recent[cand['act'], cand['slot']] = (~cand['starter'].astype(bool)) & (cand['minutes_entered'] < 10)
    target_in = pd.read_parquet(source / 'targets_in.parquet')
    if not np.array_equal(target_in.event_id, meta.event_id):
        raise ValueError('target query rows misaligned')
    query = np.column_stack([meta.period, meta.sec / 3600, target_in.x / 100, target_in.y / 100]).astype(np.float32)
    np.savez(out / 'evaluation.npz', cand_player=raw, cand_mask=mask, prior_matches=prior,
             recent_substitute=recent, steps=target_in.steps_code.to_numpy(),
             actor_slot=np.load(source / 'arrays_label.npz')['actor_slot'], query=query)
    meta['block'] = meta.block.map({'model_train': 0, 'model_validation': 1}).astype(np.int8)
    meta['league'] = dc['league']
    meta['kickoff'] = meta.match_id.map(matches.set_index('match_id').kickoff)
    meta.to_parquet(out / 'targets.parquet', index=False)
    matches.to_parquet(out / 'matches.parquet', index=False)
    # Written last: the loader requires this completion marker and the exact schema.
    manifest = {'scope': spec['scope'], 'windows': windows, 'targets': len(meta),
                'query': ['period', 'period_seconds/3600', 'start_x/100', 'start_y/100'],
                'native_fields': {f: list(v) for f, v in empty.items()}, 'task': cfg['task']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--dataset', type=int, required=True)
    a = p.parse_args()
    print(prepare(load(ROOT / a.config), a.dataset))


if __name__ == '__main__':
    main()
