"""Check an explicit action/candidate dataset against its manifest and exact on-pitch rules."""
import argparse
import json

import numpy as np
import pandas as pd

from common import ROOT, load
from event_schema import PERIOD_ORDER, ROLES


def validate(cfg, test_manifest):
    A = pd.read_parquet(ROOT / cfg['actions_path'])
    onpitch = pd.read_parquet(ROOT / cfg['onpitch_path'])
    selected = {r['match_id'] for r in load(ROOT / cfg['train_manifest'])}
    tests = {r['match_id'] for r in load(ROOT / test_manifest)}
    actual = set(A['match_id'])
    if actual != selected or actual & tests:
        raise AssertionError('Action matches differ from the selected non-test manifest')
    if A['event_id'].duplicated().any():
        raise AssertionError('Duplicate target event identifiers')
    with np.load(ROOT / cfg['who_path']) as arrays:
        W = dict(arrays)
    if len({len(v) for v in W.values()}) != 1:
        raise AssertionError('Candidate array lengths differ')
    act, cand = W['act'], W['cand']
    if ((act < 0) | (act >= len(A))).any():
        raise AssertionError('Invalid action index')
    if pd.MultiIndex.from_arrays([act, cand]).duplicated().any():
        raise AssertionError('Duplicate candidate in an action')
    if not np.array_equal(W['y'], (A['player_id'].to_numpy()[act] == cand).astype(float)):
        raise AssertionError('Candidate outcomes do not identify the actor')
    actual_counts = np.bincount(act, minlength=len(A))
    actual_positive = np.bincount(act, weights=W['y'], minlength=len(A))
    if (actual_positive[actual_counts > 0] != 1).any():
        raise AssertionError('Each retained action must have exactly one actor')
    spans = {k:g for k,g in onpitch.groupby(['match_id', 'team_id'])}
    expected_counts = np.zeros(len(A), dtype=int)
    actor_inside = np.zeros(len(A), dtype=bool)
    tolerance_rows = 0
    records_with_cards = 0
    for (match, team), idx in A.groupby(['match_id', 'team_id']).indices.items():
        rows, players = A.iloc[idx], spans[(match, team)]
        minute = rows['minute'].to_numpy()[:, None]
        period = rows['period'].map(PERIOD_ORDER).to_numpy()[:, None]
        second = rows['event_sec'].to_numpy()[:, None]
        cp, cs = players['card_period'].to_numpy(), players['card_sec'].to_numpy()
        before = (cp < 0) | (period < cp) | ((period == cp) & (second < cs))
        tol = cfg['onpitch_tolerance_minutes']
        start, end = players['start'].to_numpy(), players['sub_end'].to_numpy()
        on = (minute >= start-tol) & (minute < end+tol) & before
        strict = (minute >= start) & (minute < end) & before
        expected_counts[idx] = on.sum(1)
        actor_inside[idx] = (on & (rows['player_id'].to_numpy()[:, None] == players['player_id'].to_numpy())).any(1)
        tolerance_rows += int((on & ~strict & actor_inside[idx, None]).sum())
        records_with_cards += int((cp >= 0).sum())
        for role in ROLES:
            expected = on[:, players['role'].to_numpy() == role].sum(1)
            if not np.array_equal(expected, rows[f'n_on_{role}'].to_numpy()):
                raise AssertionError(f'On-pitch role counts differ in match {match}, team {team}')
        # Check candidate identities, not only their total count.
        candidate_rows = np.flatnonzero((W['match'] == match) & np.isin(act, idx))
        local = {a:i for i,a in enumerate(idx)}
        player_column = {p:j for j,p in enumerate(players['player_id'])}
        for r in candidate_rows:
            if cand[r] not in player_column or not on[local[act[r]], player_column[cand[r]]]:
                raise AssertionError('Candidate is not on pitch')
    if not np.array_equal(actual_counts, expected_counts * actor_inside):
        raise AssertionError('Candidate coverage differs from the on-pitch set')
    pairs = pd.read_parquet(ROOT / cfg['coinciding_path'])
    if len(pairs) and (not set(pairs['target_event_id']) <= set(A['event_id']) or
                       not pairs['dt'].between(0, cfg['coincide_seconds']).all()):
        raise AssertionError('Invalid overlap-pair target or time window')
    result = {'league':cfg['league'], 'matches':len(actual), 'actions':len(A), 'test_overlap':len(actual & tests),
              'candidate_rows':len(act), 'actions_in_candidates':int((actual_counts > 0).sum()),
              'excluded_actor_actions':int((~actor_inside).sum()), 'tolerance_candidate_rows':tolerance_rows,
              'red_card_player_matches':records_with_cards, 'onpitch_count_mismatches':0,
              'actions_with_opponent_overlap':int((A['n_coinciding_opponent'] > 0).sum()),
              'actions_with_own_overlap':int((A['n_coinciding_own'] > 0).sum())}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--test-manifest', required=True)
    p.add_argument('--out')
    args = p.parse_args()
    result = validate(load(args.config), args.test_manifest)
    text = json.dumps(result, indent=2) + '\n'
    if args.out:
        (ROOT / args.out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
