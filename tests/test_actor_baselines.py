"""Actor adapters: causality, training-only transforms, cold output, source cores and metrics."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from actor_baselines import ActorBaseline, FAMILIES, Seq2EventHistory, NMSTPPHistory, LEMHistory, positional_encoding
from actor_benchmark_data import batch, encode_ids, fit_preprocessing, transform, validate_temporal_split
from actor_benchmark_metrics import check_alignment, paired_comparison, score_archive
from actor_native_inputs import NativeState, match_views, CENTERS
from train_actor_baseline import train
from plan_actor_baselines import select, run_path

CFG = json.loads((ROOT / 'configs/actor_baselines/england.json').read_text())
WINDOWS = {k: v['window'] for k, v in CFG['encoders'].items()}
torch.set_num_threads(1)
torch.backends.mha.set_fastpath_enabled(False)


def event(i, team=1, name='Pass', sub='Simple pass', tags=(), x=50, period='1H'):
    return dict(id=i, teamId=team, eventName=name, subEventName=sub, tags=[{'id': t} for t in tags],
                positions=[dict(x=x, y=40)], eventSec=i * 3.3, matchPeriod=period)


def native_equal(test, a, b):
    for family in FAMILIES:
        for key in a[family]:
            np.testing.assert_array_equal(a[family][key], b[family][key])


def synthetic(family, n=160):
    state = NativeState(1, 2, {})
    views = []
    for i in range(n):
        views.append(state.view(1, WINDOWS)[family])
        state.consume(event(i + 1, team=1 + i % 2, x=10 + i % 80))
    X = {k: np.stack([v[k] for v in views]) for k in views[0]}
    X.update(query=np.tile([0, .3, .5, .4], (n, 1)).astype('float32'),
             cand_player=np.tile([11, 22, 0], (n, 1)), cand_mask=np.tile([True, True, False], (n, 1)),
             prior_matches=np.tile([0, 6, 0], (n, 1)), recent_substitute=np.zeros((n, 3), bool),
             steps=np.zeros(n, int))
    X['cand_player'][128:, 1] = 99
    actor = np.arange(n) % 2
    meta = pd.DataFrame({'match_id': np.repeat([1, 2, 3, 4, 5], 32), 'seq': np.tile(np.arange(32), 5),
                         'event_id': np.arange(n), 'block': (np.arange(n) >= 128).astype(int),
                         'league': 'England', 'kickoff': np.repeat([1, 2, 3, 4, 5], 32), 'minute': 5.})
    return X, actor, meta


class NativeInputs(unittest.TestCase):
    def test_target_future_and_player_ids_do_not_enter_history(self):
        events = [event(i) for i in range(6)]
        targets = [(0, 3, 1)]
        a = dict(match_views(events, targets, {}, 1, 2, {}, WINDOWS))[0]
        changed = copy.deepcopy(events)
        for e in changed:
            e['playerId'] = 999999
        changed[3:] = [dict(deliberately_invalid_target=True)] * 3
        b = dict(match_views(changed, targets, {}, 1, 2, {}, WINDOWS))[0]
        native_equal(self, a, b)

    def test_exclusion_precedes_score_delta_and_possession(self):
        events = [event(0), event(1, team=2, name='Shot', sub='Shot', tags=[101, 1701]), event(2, x=20), event(3)]
        a = dict(match_views(events, [(0, 3, 1)], {0: {1}}, 1, 2, {}, WINDOWS))[0]
        state = NativeState(1, 2, {})
        state.consume(events[0]); state.consume(events[2])
        native_equal(self, a, state.view(1, WINDOWS))
        np.testing.assert_array_equal(a['lem']['tabular'][9:15], 0)
        mutated = copy.deepcopy(events)
        mutated[1] = event(1, name='Pass', sub='Cross', tags=[102], x=99)
        native_equal(self, a, dict(match_views(mutated, [(0, 3, 1)], {0: {1}}, 1, 2, {}, WINDOWS))[0])

    def test_marker_requires_observed_possession_change(self):
        state = NativeState(1, 2, {})
        state.consume(event(1))
        self.assertEqual(state.view(1, WINDOWS)['seq2event']['actions'][-1], 4)
        state.consume(event(2, team=2))
        self.assertEqual(state.view(1, WINDOWS)['seq2event']['actions'][-1], 1)
        self.assertEqual(state.view(1, WINDOWS)['nmstpp']['actions'][-2], 1)

    def test_nm_zone_is_native_nearest_centroid(self):
        state = NativeState(1, 2, {})
        e = event(1, x=float(CENTERS[13, 0])); e['positions'][0]['y'] = float(CENTERS[13, 1])
        state.consume(e)
        self.assertEqual(state.view(1, WINDOWS)['nmstpp']['zones'][-1], 13)

    def test_training_only_transforms_and_allowlist(self):
        for family in FAMILIES:
            X, _, _ = synthetic(family)
            pre = fit_preprocessing(X, np.arange(128), 4, family)
            self.assertEqual(pre['player_ids'], [11, 22])
            x = transform(X, pre)
            self.assertEqual(x['cand_id'][129, 1], 0)
            self.assertFalse(set(x) & {'prior_matches', 'recent_substitute', 'steps', 'cand_player', 'actor_slot'})
            changed = copy.deepcopy(X)
            changed['prior_matches'][:] = 999
            changed['recent_substitute'][:] = True
            for k, v in transform(changed, pre).items():
                np.testing.assert_array_equal(v, x[k])
            if family != 'lem':
                changed['continuous'][128:] *= 1000
            self.assertEqual(fit_preprocessing(changed, np.arange(128), 4, family), pre)
        np.testing.assert_array_equal(encode_ids(np.array([11, 99, 0]), [11, 22]), [1, 0, 0])

    def test_global_cut_rejects_overlapping_leagues(self):
        matches = pd.DataFrame({'match_id': [1, 2, 3, 4], 'kickoff': [1, 3, 4, 5],
                                 'block': ['model_train', 'model_validation', 'model_train', 'model_validation']})
        with self.assertRaises(ValueError):
            validate_temporal_split(matches)


class Models(unittest.TestCase):
    def test_padding_batch_candidate_permutation_gradients_and_query(self):
        for family in FAMILIES:
            X, _, _ = synthetic(family)
            pre = fit_preprocessing(X, np.arange(128), 4, family)
            inputs = transform(X, pre)
            b = batch(inputs, np.array([0, 5, 129]), 'cpu')
            model = ActorBaseline(family, CFG, 2).eval()
            score, _ = model(b)
            self.assertTrue(torch.isfinite(score[:, :2]).all())
            torch.testing.assert_close(model({k: v[[2, 0, 1]] for k, v in b.items()})[0], score[[2, 0, 1]])
            reordered = dict(b)
            for k in ('cand_id', 'cand_mask'):
                reordered[k] = b[k][:, [1, 0, 2]]
            torch.testing.assert_close(model(reordered)[0], score[:, [1, 0, 2]])
            if family != 'lem':
                bad = {k: v.clone() for k, v in b.items()}
                bad['actions'][~bad['valid']] = 9999
                bad['continuous'][~bad['valid']] = float('nan')
                if family == 'nmstpp':
                    bad['zones'][~bad['valid']] = 9999
                torch.testing.assert_close(model(bad)[0], score)
            q = dict(b); q['query'] = b['query'].clone().requires_grad_(True)
            loss = -model(q)[0].log_softmax(1)[:, 0].mean()
            loss.backward()
            self.assertGreater(float(q['query'].grad.abs().sum()), 0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            q['cand_id'] = torch.zeros_like(b['cand_id'])
            unseen = model(q)[0]
            torch.testing.assert_close(unseen[:, 0], unseen[:, 1])

    def test_upstream_seq2event_core_equality(self):
        path = ROOT / 'external/baselines/Soccer-Seq2Event/Seq2Event_Notebook02_Modelling.ipynb'
        cells = json.loads(path.read_text())['cells']
        definitions = []
        for name in ('ian_generate_positional_encoding', 'Soccer_Model_3ze'):
            src = next(''.join(c['source']) for c in cells if f'def {name}' in ''.join(c['source']) or f'class {name}' in ''.join(c['source']))
            definitions += [node for node in ast.parse(src).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
        ns = dict(torch=torch, nn=nn, F=F, np=np, device='cpu', num_action_cats=7,
                  cat_embedding_dim=7, act_padding_idx=1, scale_grad_by_freq=True, num_contvars_in=10,
                  cont_embedding_dim=10, model_type='Transformer', d_model=17, transformer_nhead=1,
                  dim_feedforward=8, num_layers=1, transformer_finaldenselayer_dim=64, num_contvars_out=2)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), ns)
        with contextlib.redirect_stdout(io.StringIO()):
            original = ns['Soccer_Model_3ze']().eval()
        original.lin2 = nn.Identity()
        port = Seq2EventHistory(CFG['encoders']['seq2event']).eval()
        port.load_state_dict({k: v for k, v in original.state_dict().items() if not k.startswith('encoder_layer.')})
        b = dict(actions=torch.randint(0, 7, (3, 40)), continuous=torch.rand(3, 40, 10), valid=torch.ones(3, 40, dtype=torch.bool))
        x = torch.cat([b['actions'][..., None], b['continuous']], -1)
        torch.testing.assert_close(original(x), port(b), atol=2e-6, rtol=2e-5)

    def test_upstream_lem_core_equality(self):
        path = ROOT / 'external/baselines/Unified-LEM/0001 Train Tabular LEMs.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'MLP')
        ns = dict(torch=torch, nn=nn, List=List)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        original = ns['MLP'](33, [196] * 3, 101, .3).eval()
        original.model[-1] = nn.Identity()
        port = LEMHistory(CFG['encoders']['lem']).eval()
        port.load_state_dict(original.state_dict())
        x = torch.randn(4, 33)
        torch.testing.assert_close(original(x), port({'tabular': x}))

    def test_upstream_nmstpp_core_after_sequence_axis_fix(self):
        path = ROOT / 'external/baselines/NMSTPP/NMSTPP.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'NMSTPP')
        ns = dict(torch=torch, nn=nn, device='cpu', action_emb_in=5, action_emb_out=5, zone_emb_in=20,
                  zone_emb_out=20, other_lin_in=6, other_lin_out=6, input_features_len=31,
                  mutihead_attention=1, hidden_dim=1024, scale_grad_by_freq=True)
        # Isolate the class definition; never execute upstream dataset loading/training.
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        with contextlib.redirect_stdout(io.StringIO()):
            original = ns['NMSTPP']().eval()
        port = NMSTPPHistory(CFG['encoders']['nmstpp']).eval()
        port.load_state_dict({k: v for k, v in original.state_dict().items() if k in port.state_dict()})
        b = dict(actions=torch.randint(0, 5, (3, 40)), zones=torch.randint(0, 20, (3, 40)),
                 continuous=torch.rand(3, 40, 6), valid=torch.ones(3, 40, dtype=torch.bool))
        x = torch.cat([original.emb_act(b['actions']), original.emb_zone(b['zones']), original.lin0(b['continuous'])], -1)
        x += positional_encoding(40, 31, 'cpu', x.dtype)[None]
        expected = original.lin_relu(original.encoder_layer(x)[:, -1])
        torch.testing.assert_close(expected, port(b))

    def test_all_training_checkpoint_reload_and_metrics(self):
        cfg = copy.deepcopy(CFG)
        cfg['smoke']['validation_targets'] = 16
        for family in FAMILIES:
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                out = Path(d) / family
                result = train(cfg, family, 20261021, .001, 'smoke', torch.device('cpu'), out, synthetic(family))
                self.assertTrue(np.isfinite(result['best_validation_nll']))
                saved = torch.load(out / 'best.pt', weights_only=True)
                model = ActorBaseline(family, cfg, len(saved['preprocessing']['player_ids']))
                model.load_state_dict(saved['state_dict'])
                archive = dict(np.load(out / 'predictions.npz'))
                self.assertEqual(len(archive['scores']), 16)
                self.assertTrue((out / 'per_match.csv').exists())


class Metrics(unittest.TestCase):
    def test_lr_selection_requires_complete_matching_search(self):
        X, actor, meta = synthetic('lem')
        from actor_benchmark_data import prediction_archive
        pre = fit_preprocessing(X, np.arange(128), 4, 'lem')
        rows = np.arange(128, 160)
        scores = np.tile([0., 0., -np.inf], (len(rows), 1))
        archive = prediction_archive(X, actor, meta, rows, scores, pre)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for family in FAMILIES:
                for lr in CFG['learning_rates']:
                    path = run_path(root, family, CFG['search_seed'], lr)
                    path.mkdir(parents=True)
                    run = dict(mode='train', family=family, seed=CFG['search_seed'], lr=lr,
                               epochs=CFG['training']['epochs'], optimizer_steps=100,
                               best_validation_nll=1.0 if lr == .001 else 2.0)
                    (path / 'run.json').write_text(json.dumps(run))
                    (path / 'config.json').write_text(json.dumps(CFG))
                    np.savez(path / 'predictions.npz', **archive)
                    np.savez(path / 'training_rows.npz', event_id=np.arange(128))
            self.assertTrue(all(v['lr'] == .001 for v in select(CFG, root).values()))
            (path / 'run.json').write_text(json.dumps({**run, 'mode': 'smoke'}))
            with self.assertRaises(ValueError):
                select(CFG, root)

    def test_ties_unreachable_single_candidate_and_pairing(self):
        a = dict(scores=np.array([[0., 0., -np.inf], [0., 0., -np.inf], [0., -np.inf, -np.inf]]),
                 candidate_mask=np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]], bool),
                 candidate_player=np.array([[20, 10, 0], [20, 10, 0], [20, 0, 0]]),
                 actor_slot=np.array([0, -1, 0]), match_id=np.array([1, 1, 2]), seq=np.array([1, 2, 1]),
                 event_id=np.array([1, 2, 3]), league=np.array(['England'] * 3),
                 prior_matches=np.full((3, 3), 10), known_in_training=np.ones((3, 3), bool),
                 recent_substitute=np.zeros((3, 3), bool), minute=np.zeros(3), steps=np.zeros(3))
        metrics, table = score_archive(a)
        self.assertAlmostEqual(metrics['actions']['all']['top1'], .5)
        self.assertAlmostEqual(metrics['actions']['all']['nll'], np.log(2) / 2)
        self.assertIsNone(metrics['candidates']['history_lt5']['binary_log_loss'])
        compare = paired_comparison(table, table, 100, 4)
        self.assertEqual(compare['all/nll']['ci95'], [0., 0.])
        changed = copy.deepcopy(a); changed['candidate_player'][0, 0] = 99
        self.assertEqual(score_archive(changed)[0], metrics)
        with self.assertRaises(ValueError):
            check_alignment(a, changed)

    def test_expected_ties_match_enumerated_permutations(self):
        from itertools import permutations
        from ranking_metrics import expected_topk
        # One higher candidate, three tied candidates, then padding.
        scores = np.array([[2., 1., 1., 1., -np.inf]])
        ranks = [[0, *order] for order in permutations([1, 2, 3])]
        hits = expected_topk(scores, np.array([2]), ks=(1, 2, 3, 4))
        for k, value in hits.items():
            self.assertAlmostEqual(value[0], np.mean([2 in order[:k] for order in ranks]))
        np.testing.assert_array_equal(expected_topk(scores, np.array([-1]))[1], 0)


if __name__ == '__main__':
    unittest.main()
