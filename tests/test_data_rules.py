import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from build_actions import acquisition_level, onpitch_intervals, validate_card_overrides
from build_candidates import expand_candidates
from numeric import holm, newton, loglik
from pooling import forward_pool


class DataRules(unittest.TestCase):
    def setUp(self):
        self.players = pd.DataFrame([
            dict(match_id=1, team_id=1, player_id=10, start=0., sub_end=np.inf, card_period=0, card_sec=60.),
            dict(match_id=1, team_id=1, player_id=20, start=0., sub_end=np.inf, card_period=-1, card_sec=np.nan),
            dict(match_id=1, team_id=1, player_id=30, start=2., sub_end=np.inf, card_period=-1, card_sec=np.nan),
        ])

    def action(self, sec, actor=20):
        return dict(match_id=1, team_id=1, minute=sec/60, period='1H', event_sec=sec, player_id=actor)

    def test_red_card_exact_boundary_and_substitution_tolerance(self):
        A = pd.DataFrame([self.action(59), self.action(60), self.action(61)])
        W, counts = expand_candidates(A, self.players, 1., 200)
        groups = [set(W['cand'][W['act'] == a]) for a in range(3)]
        self.assertEqual(groups, [{10,20}, {20,30}, {20,30}])
        self.assertEqual(counts['rows_on_by_tolerance_only'], 2)

    def test_candidate_identity_does_not_follow_actor(self):
        A = pd.DataFrame([self.action(59, 10)])
        B = pd.DataFrame([self.action(59, 20)])
        a, _ = expand_candidates(A, self.players, 1., 200)
        b, _ = expand_candidates(B, self.players, 1., 200)
        np.testing.assert_array_equal(a['cand'], b['cand'])

    def test_actor_outside_is_excluded(self):
        A = pd.DataFrame([self.action(61, 10), self.action(62, 999)])
        W, counts = expand_candidates(A, self.players, 1., 200)
        self.assertEqual(len(W['act']), 0)
        self.assertEqual(counts['actions_without_actor_in_set'], 2)

    def test_future_action_does_not_change_current_state(self):
        A = pd.DataFrame([self.action(30,10), self.action(40,20), self.action(45,10)])
        all_rows, _ = expand_candidates(A, self.players, 1., 200)
        prefix, _ = expand_candidates(A.iloc[:2], self.players, 1., 200)
        keep = all_rows['act'] < 2
        for key in prefix:
            np.testing.assert_array_equal(prefix[key], all_rows[key][keep])

    def test_first_action_tag_is_not_predictive_input(self):
        first = {'eventName':'Others on the ball','subEventName':'Touch','tags':[{'id':1401}]}
        prev = {'eventName':'Pass','subEventName':'Simple pass'}
        self.assertEqual(acquisition_level(first, [], prev, 1, False), 'opp_pass_lost')
        self.assertEqual(acquisition_level(first, [], prev, 1, True), 'interception')

    def test_second_yellow_ends_interval(self):
        match = {'wyId':1,'teamsData':{'1':{'formation':{'lineup':[{'playerId':10,'redCards':'1'}],
                  'bench':[],'substitutions':[]}}}}
        e = {'playerId':10,'tags':[{'id':1703}],'matchPeriod':'1H','eventSec':60.}
        rows, unlisted = onpitch_intervals(match, {10:'DF'}, [e], set())
        self.assertEqual((rows[0]['card_period'], rows[0]['card_sec']), (0,60.))
        self.assertEqual(unlisted, 0)

    def test_explicit_dismissal_requires_matching_event_and_player(self):
        match = {'wyId':1,'teamsData':{'1':{'formation':{'lineup':[{'playerId':10,'redCards':'1'}],
                  'bench':[],'substitutions':[]}}}}
        e = {'id':99,'playerId':10,'tags':[{'id':1702}],'matchPeriod':'2H','eventSec':1452.}
        with self.assertRaises(ValueError):
            onpitch_intervals(match, {10:'DF'}, [e], set())
        override = {'match_id':1, 'player_id':10, 'event_id':99}
        fixed = validate_card_overrides({1:[e]}, [override])
        rows, _ = onpitch_intervals(match, {10:'DF'}, [e], fixed)
        self.assertEqual((rows[0]['card_period'], rows[0]['card_sec']), (1,1452.))
        with self.assertRaises(ValueError):
            validate_card_overrides({1:[e]}, [{**override, 'player_id':20}])
        with self.assertRaises(ValueError):
            validate_card_overrides({1:[]}, [override])


class StatisticsRules(unittest.TestCase):
    def test_holm_preserves_input_order(self):
        np.testing.assert_array_equal(holm(np.array([.04,.001,.02]), .05), [True,True,True])
        np.testing.assert_array_equal(holm(np.array([.04,.03,.001]), .05), [False,False,True])

    def test_ridge_newton_matches_independent_optimizer(self):
        from scipy.optimize import minimize
        rng = np.random.default_rng(42)
        group = np.repeat(np.arange(4),100)
        offset = rng.normal(0,.5,len(group))
        y = rng.binomial(1, .4, len(group))
        alpha = 3.
        got = newton(offset,y,group,4,alpha)
        optimum = minimize(lambda b:-loglik(offset+b[group], y).sum()+alpha*np.sum(b*b)/2,
                           np.zeros(4), method='BFGS', options={'gtol':1e-6})
        np.testing.assert_allclose(got,optimum.x,atol=1e-5)

    def test_pooling_weights_by_row_count(self):
        data=[{'sum':np.array([1.,2.]),'n':np.array([10,20])},
              {'sum':np.array([3.,4.]),'n':np.array([30,40])}]
        result=forward_pool(data,100,np.random.default_rng(42))
        self.assertAlmostEqual(result['gain'],100.)
        np.testing.assert_allclose(result['ci'],[100.,100.])


if __name__ == '__main__':
    unittest.main()
