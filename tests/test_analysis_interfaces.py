"""Exercise the complete pooling interfaces with explicitly synthetic inputs."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pool_existence import GLM_ITEMS, VC_ITEMS


class AnalysisInterfaces(unittest.TestCase):
    def test_learnability_requires_current_existence(self):
        names = ['T2a team x steps (long_pass)', 'T2c team x steps (back_pass)', 'who|game_state_form']
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            results = {}
            for league in ['England', 'France']:
                path = base/league
                path.mkdir()
                results[league] = str(path)
                for part in ['population', 'ladder']:
                    group = names[-1:] if part == 'population' else names[:2]
                    arrays = {}
                    for name in group:
                        arrays[f'{name}|sum'] = np.array([1.,2.])
                        arrays[f'{name}|n'] = np.array([10,20])
                    np.savez(path/f'{part}.npz', **arrays)
                (path/'ladder.json').write_text(json.dumps({'interactions':{}}))
            (base/'existence.json').write_text(json.dumps({'items':{'T2a':{'confirmed':False},
                            'T2c':{'confirmed':True},'R1-Pa':{'confirmed':False}}}))
            config = base/'config.json'
            config.write_text(json.dumps({'leagues':list(results),'results_dirs':results,'bootstrap':32,'seed':42,
                    'existence_results':str(base/'existence.json'),
                    'out_json':str(base/'pooled.json'),'out_md':str(base/'pooled.md')}))
            subprocess.run([sys.executable,str(ROOT/'src/pool_learnability.py'),'--config',str(config)],
                           check=True,capture_output=True,text=True)
            decisions = json.loads((base/'pooled.json').read_text())['decisions']
            self.assertEqual(decisions[names[0]]['decision'], '候选，默认关闭')
            self.assertEqual(decisions[names[1]]['decision'], '显式')
            self.assertEqual(decisions[names[2]]['decision'], '候选，默认关闭')

    def test_pooling_and_eligibility(self):
        rng = np.random.default_rng(10)
        leagues = ['France', 'Germany', 'Italy', 'Spain']
        who_vc = {'R1-Ea','R1-Eb','R3-Ea','R3-Eb','R4-Ea','R4-Eb','T1','I1a','I1b','C2'}
        who_glm = {'R1-Pa','R3-P','R4-P'}
        families = json.loads((ROOT/'configs/families.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            results = {}
            roles = ['a','b','c']
            profile = [0., .2, .4]
            (base/'reference.json').write_text(json.dumps({'glm_beta':{n:[1.] for n in GLM_ITEMS},
                                                         'p1_roles':roles,'p1_profile':profile}))
            # The interface must never promote an ineligible hypothesis.
            (base/'eligibility.json').write_text(json.dumps({'eligible':{n:False for f in families['families'].values() for n in f}}))
            for league in leagues:
                path = base/league
                path.mkdir()
                results[league] = str(path)
                def glm():
                    return {'names':['effect'],'beta':[.5],'cov':[[.01]],'wald':25.,'df':1,'p':1e-6,'n':100}
                def vc():
                    return {'T':30.,'V':100.,'kept':5,'T_level':[6.]*5,'V_level':[20.]*5,
                            'perm_level_mean':[2.]*5}
                acts = [[10.,10.,10.,10.,10.], [10.,10.,10.,10.,12.], [10.,10.,10.,10.,15.]]
                who = {'league':league,'rows':{'candidates_all':100,'with_prior_role':90},
                       'glm':{n:glm() for n in who_glm},'vc':{n:vc() for n in who_vc|{'TY1|none','TY1|type'}},
                       'p1':{'roles':roles,'acts':acts,'opportunities':[[100.]*5]*3}}
                event = {'league':league,'matches':4,'teams':2,'rows':{'team':40},
                         'glm':{n:glm() for n in set(GLM_ITEMS)-who_glm},'vc':{n:vc() for n in set(VC_ITEMS)-who_vc},
                         'p3prime':{'glm':glm(),'glm_control_opp_pass_lost':glm(),
                                    'box_shot':{'active':[2,10],'control':[1,10]},'counts':{}},
                         'descriptive_B':{'high_regain':[4,10],'other_first_actions':[3,10]}}
                wa,ea={},{}
                for n in VC_ITEMS+['TY1|none','TY1|type']:
                    target=wa if n in who_vc or n.startswith('TY1|') else ea
                    target[f'perm|{n}']=rng.normal(10,1,20)
                for kind in ['none','type']:
                    wa[f'ratio|{kind}|T']=np.full(20,30. if kind=='none' else 15.)
                    wa[f'ratio|{kind}|null']=np.full(20,10.)
                    wa[f'ratio|{kind}|V']=np.full(20,100.)
                for n in ['TY3a','TY4','TY1a','TY1b','TY1_individual','TY2','TY2_role','TY2_physical']:
                    target=ea if n=='TY3a' else wa
                    target[f'fwd|{n}|sum']=np.array([.1,.2,.3,.4])
                    target[f'fwd|{n}|n']=np.array([10,10,10,10])
                ea['mh_tables']=np.array([[3,7,2,8],[4,6,3,7]])
                for name,obj in [('who',who),('event',event)]:
                    (path/f'{name}.json').write_text(json.dumps(obj))
                np.savez(path/'who.npz',**wa)
                np.savez(path/'event.npz',**ea)
            cfg={'leagues':leagues,'results_dirs':results,'family_json':str(ROOT/'configs/families.json'),
                 'reference_json':str(base/'reference.json'),'eligibility_json':str(base/'eligibility.json'),
                 'seed':42,'bootstrap':32,'p1_permutations':32,'p1_threshold':.8,'ty1_ratio_max':.8,
                 'out_json':str(base/'pooled.json'),'out_md':str(base/'pooled.md')}
            config=base/'config.json';config.write_text(json.dumps(cfg))
            subprocess.run([sys.executable,str(ROOT/'src/pool_existence.py'),'--config',str(config)],
                           check=True,capture_output=True,text=True)
            result=json.loads((base/'pooled.json').read_text())
            self.assertTrue(all(not r.get('confirmed',False) for r in result['items'].values()))
            self.assertEqual(set(result['families']['main']),set(families['families']['main']))


if __name__ == '__main__':
    unittest.main()
