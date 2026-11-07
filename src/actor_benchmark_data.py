"""Load native caches; fit transforms and the actor output vocabulary on training only.

Evaluation-only history counts never enter transform() or batch().
"""
import numpy as np
import pandas as pd
import torch

from common import ROOT, load

KEYS = ('match_id', 'seq', 'event_id')
NATIVE_KEYS = {'seq2event': ('actions', 'continuous', 'valid'),
               'nmstpp': ('actions', 'zones', 'continuous', 'valid'), 'lem': ('tabular',)}


def validate_temporal_split(matches):
    if matches.match_id.duplicated().any():
        raise ValueError('a match occurs in more than one scope')
    train = matches.loc[matches.block == 'model_train', 'kickoff']
    val = matches.loc[matches.block == 'model_validation', 'kickoff']
    if not len(train) or not len(val) or train.max() >= val.min():
        raise ValueError('all training matches across ALL scopes must precede every validation match')


def read_data(cfg, family, require_training_cut):
    arrays, tables, matches = [], [], []
    for spec in cfg['datasets']:
        d = ROOT / spec['native_dir']
        manifest = load(d / 'manifest.json')
        windows = {k: v['window'] for k, v in cfg['encoders'].items()}
        if manifest['scope'] != spec['scope'] or manifest['windows'] != windows or manifest['task'] != cfg['task']:
            raise ValueError('native cache differs from the explicit scope/task/window configuration')
        a = dict(np.load(d / 'evaluation.npz'))
        for k in NATIVE_KEYS[family]:
            a[k] = np.load(d / f'{family}_{k}.npy', mmap_mode='r')
        tm, match = pd.read_parquet(d / 'targets.parquet'), pd.read_parquet(d / 'matches.parquet')
        scope = load(ROOT / spec['scope'])
        for block, key in [('model_train', 'train_manifest'), ('model_validation', 'validation_manifest')]:
            expected = {r['match_id']: r['kickoff'] for r in load(ROOT / scope[key])}
            if match.loc[match.block == block].set_index('match_id').kickoff.to_dict() != expected:
                raise ValueError(f'cache match table differs from {key}')
        test = {r['match_id'] for r in load(ROOT / scope['test_manifest'])}
        if set(tm.match_id) & test or set(tm.match_id) != set(match.match_id):
            raise ValueError('invalid development match set')
        if any(len(v) != len(tm) for v in a.values()) or len(tm) != manifest['targets']:
            raise ValueError('native cache row counts disagree')
        arrays.append(a); tables.append(tm); matches.append(match)
    meta = pd.concat(tables, ignore_index=True)
    matches = pd.concat(matches, ignore_index=True)
    if require_training_cut:
        validate_temporal_split(matches)
    if meta.duplicated(list(KEYS[:2])).any():
        raise ValueError('duplicate targets')
    X = arrays[0] if len(arrays) == 1 else {k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    actor = X.pop('actor_slot')
    mask = X['cand_mask']
    if not mask.any(1).all() or (actor < -1).any() or (actor >= mask.shape[1]).any():
        raise ValueError('invalid candidate set or actor slot')
    reach = actor >= 0
    if not mask[np.flatnonzero(reach), actor[reach]].all() or (X['cand_player'][mask] <= 0).any():
        raise ValueError('invalid reachable actor or candidate player ID')
    return X, actor, meta


def encode_ids(raw, player_ids):
    vocab = np.asarray(player_ids, dtype=np.int64)
    if not len(vocab) or not (np.diff(vocab) > 0).all():
        raise ValueError('training player vocabulary must be sorted, unique and nonempty')
    pos = np.searchsorted(vocab, raw)
    known = (pos < len(vocab)) & (vocab[np.minimum(pos, len(vocab) - 1)] == raw)
    return np.where(known, pos + 1, 0).astype(np.int64)


def rank_reference(values):
    unique, counts = np.unique(values, return_counts=True)
    return {'values': unique.tolist(), 'counts': counts.tolist()}


def percentile(values, reference):
    ref = np.asarray(reference['values'])
    counts = np.asarray(reference['counts'])
    cumulative = np.r_[0, counts.cumsum()]
    left, right = np.searchsorted(ref, values, side='left'), np.searchsorted(ref, values, side='right')
    # Pandas rank(pct=True), including average ranks for observed ties.
    return (cumulative[left] + cumulative[right] + (left != right)) / (2 * cumulative[-1])


def fit_preprocessing(X, fit_rows, train_end, family):
    mask = X['cand_mask'][fit_rows]
    players = np.unique(X['cand_player'][fit_rows][mask])
    if not len(players):
        raise ValueError('empty training player vocabulary')
    pre = {'family': family, 'player_ids': players.tolist(), 'train_last_kickoff': int(train_end),
           'unseen_actor_rule': 'mean_trained_class_logit'}
    if family != 'lem':
        valid = X['valid'][fit_rows]
        num = X['continuous'][fit_rows][valid].astype(np.float64)
        if not len(num) or not np.isfinite(num).all():
            raise ValueError('empty or non-finite native training features')
        if family == 'seq2event':
            pre['gap_cap'] = .01 * float(num[:, 0].max())
            num[:, 6] = np.clip(num[:, 6], 0, pre['gap_cap'])
            pre['ranks'] = {str(c): rank_reference(num[:, c]) for c in (0, 5, 6)}
            columns = [4, 7, 8]
        else:
            columns = list(range(6))
        lo, hi = num[:, columns].min(0), num[:, columns].max(0)
        # Standard MinMax convention for a constant training column; no validation fitting.
        pre['range_columns'], pre['low'], pre['scale'] = columns, lo.tolist(), np.where(hi > lo, hi - lo, 1).tolist()
    return pre


def transform(X, preprocessing):
    family = preprocessing['family']
    out = {k: X[k] for k in NATIVE_KEYS[family]}
    out['query'], out['cand_mask'] = X['query'], X['cand_mask']
    out['cand_id'] = encode_ids(X['cand_player'], preprocessing['player_ids'])
    out['cand_id'][~X['cand_mask']] = 0
    if family != 'lem':
        num = X['continuous'].copy()
        if family == 'seq2event':
            num[..., 6] = np.clip(num[..., 6], 0, preprocessing['gap_cap'])
            for c, reference in preprocessing['ranks'].items():
                num[..., int(c)] = percentile(num[..., int(c)], reference)
        columns = preprocessing['range_columns']
        num[..., columns] = (num[..., columns] - np.asarray(preprocessing['low'])) / np.asarray(preprocessing['scale'])
        num[~X['valid']] = 0
        out['continuous'] = num
    if any(not np.isfinite(v).all() for v in out.values()):
        raise ValueError('non-finite model input')
    return out


def batch(inputs, rows, device):
    return {k: torch.as_tensor(v[rows], device=device) for k, v in inputs.items()}


def prediction_archive(X, actor, meta, rows, scores, preprocessing):
    raw = X['cand_player'][rows]
    return {**{k: meta[k].to_numpy()[rows] for k in KEYS},
            'league': meta.league.to_numpy(dtype=str)[rows], 'scores': scores,
            'candidate_player': raw, 'candidate_mask': X['cand_mask'][rows], 'actor_slot': actor[rows],
            'prior_matches': X['prior_matches'][rows],
            'known_in_training': encode_ids(raw, preprocessing['player_ids']) > 0,
            'recent_substitute': X['recent_substitute'][rows],
            'minute': meta.minute.to_numpy()[rows], 'steps': X['steps'][rows]}
