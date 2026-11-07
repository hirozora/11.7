"""Registered event-side specifications and population contrasts."""
import numpy as np
import pandas as pd

import hypothesis_statistics as hs
import entity_types
from common import ROOT, build, standardize
from numeric import pop_offset


def population(A, cfg):
    zone, steps = A['zone'].to_numpy(), A['steps_code'].to_numpy()
    groups = pd.factorize(A['match_id'])[0]
    U = standardize(A['U'].to_numpy())
    late = (A['trailing_late'] == 'trailing_late').to_numpy().astype(float)
    all_rows = np.ones(len(A), bool)

    def fit(target, rows, X, names, tested):
        y = A[f't_{target}'].to_numpy().astype(float)[rows]
        offset = pop_offset(y, np.zeros(rows.sum(), int), zone[rows], steps[rows],
                            np.ones(rows.sum(), bool), cfg['reference_pseudocount'])
        return hs.glm_item(y, X, names, offset, groups[rows], tested)

    out = {}
    for name, target in [('R1-Pb', 'long_pass'), ('R1-Pc', 'cross')]:
        out[name] = fit(target, all_rows, np.column_stack([np.ones(len(A)), U]), ['const', 'U'], [1])
    for name, target in [('R5-P', 'cross'), ('P2', 'long_pass')]:
        out[name] = fit(target, all_rows, np.column_stack([np.ones(len(A)), late]), ['const', 'trailing_late'], [1])
    rows = steps <= 2
    lead = (A['score_state'] == 'leading').to_numpy()[rows].astype(float)
    trail = (A['score_state'] == 'trailing').to_numpy()[rows].astype(float)
    st = steps[rows]
    X = np.column_stack([np.ones(rows.sum()), lead, trail, lead*(st == 1), lead*(st == 2), trail*(st == 1), trail*(st == 2)])
    names = ['const', 'leading', 'trailing', 'leading x steps1', 'leading x steps2', 'trailing x steps1', 'trailing x steps2']
    for name, target in [('R2-Pa', 'forward'), ('R2-Pb', 'back_pass')]:
        out[name] = fit(target, rows, X, names, [3, 4, 5, 6])
    return out


def prepare(cfg):
    A, meta = build(cfg)
    with np.load(ROOT / cfg['event_offsets_path']) as z:
        offsets = {t:z[f'team_{t}'] for t in cfg['team_targets']}
    if any(len(v) != len(A) for v in offsets.values()):
        raise ValueError('Event offsets and action rows differ in length')
    hs.STATE['cfg'] = cfg
    tm = A.groupby(['match_id', 'team_id']).ngroup().to_numpy()
    labels = A.groupby(['match_id', 'team_id'])['team_i'].first().to_numpy()
    steps = A['steps_code'].to_numpy()
    phase = A['phase'].map({'first':0, 'early':1, 'settled':2}).to_numpy()
    U = standardize(A['U'].to_numpy())
    late = (A['trailing_late'] == 'trailing_late').to_numpy().astype(int)
    score = A['score_state'].map({'drawing':0, 'leading':1, 'trailing':2}).to_numpy()

    def team(target, kind='cells', level=None, L=None, x=None):
        return {'y':A[f't_{target}'].to_numpy().astype(float), 'g':offsets[target],
                'zone':A['zone'].to_numpy(), 'unit':tm, 'unit_label':labels,
                'strata':[np.arange(len(labels))], 'attr':np.zeros(len(A), int), 'n_attr':1,
                'kind':kind, 'level':level, 'L':L, 'x':x, 'k':A['k'].to_numpy(), 'match':A['match_id'].to_numpy()}

    specs = {
        'R1-Ec':team('long_pass', 'slope', x=U), 'R1-Ed':team('cross', 'slope', x=U),
        'R2-Ea':hs.subset(team('forward', level=score, L=3), steps <= 2),
        'R2-Eb':hs.subset(team('back_pass', level=score, L=3), steps <= 2),
        'R5-E':team('cross', level=late, L=2), 'T3':team('long_pass', level=late, L=2),
    }
    for name, target in [('T2a','long_pass'), ('T2b','cross'), ('T2c','back_pass'), ('T2d','forward')]:
        specs[name] = team(target, level=steps, L=5)

    # These existence tests retain the descriptive actor-role strata in the registered protocol.
    # The predictive learnability analyses use prematch actor roles instead.
    rows = np.flatnonzero((A['actor_fine'] != 'unassigned').to_numpy())
    P = A.iloc[rows]
    unit = P.groupby(['match_id', 'player_id'], sort=False).ngroup().to_numpy()
    first = P.assign(unit=unit).groupby('unit').first()
    pidx = {p:i for i,p in enumerate(sorted(P['player_id'].unique()))}
    ridx = {r:i for i,r in enumerate(meta['roles'])}
    nR = len(ridx)
    strata = ((first['team_i'].to_numpy()*nR + first['actor_fine'].map(ridx).to_numpy())*2 + first['era'].to_numpy())*2
    strata += (first['opponent_class'] == 'big_six').to_numpy().astype(int)
    for name,target in [('I2a','long_pass'), ('I2b','forward'), ('I2c','back_pass')]:
        specs[name] = {**hs.subset(team(target, level=steps, L=5), rows), 'unit':unit,
                       'unit_label':first['player_id'].map(pidx).to_numpy(),
                       'strata':[np.flatnonzero(strata == s) for s in np.unique(strata)]}
    x = A['x'].to_numpy()
    rule = (x <= 15.5) & (phase != 0) & (A['possession_start'].to_numpy() != 'interception')
    specs['C1'] = {**team('long_pass', level=rule.astype(int), L=2),
                   'style_cell':A['zone'].to_numpy()*2 + (x <= 15.5), 'n_style':18}
    K = cfg['team_type_K']
    features, types = entity_types.team_types(A, [K], cfg['seed'])
    ty = A['team_id'].map(types[K]).to_numpy()
    cells = [(ty*5 + steps, K*5)]
    s = specs['T2a']
    specs['TY3b'] = {**s, 'g':s['g'] + entity_types.fit_terms(s['g'], s['y'], np.ones(len(A), bool), cells, [], cfg['type_ridge'])}
    type_info = {'team_types':{str(int(t)):int(v) for t,v in types[K].items()},
                 'team_features':{c:{str(int(t)):float(v) for t,v in features[c].items()} for c in features.columns}}
    return {'A':A, 'meta':meta, 'specs':specs, 'ty3_cells':cells, 'types':type_info}
