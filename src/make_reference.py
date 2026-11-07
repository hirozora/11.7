"""Combine the two explicit England direction references."""
import argparse
import json
from common import ROOT, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--who', required=True)
    p.add_argument('--event', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    who, event = load(ROOT / args.who), load(ROOT / args.event)
    if set(who['glm_beta']) & set(event['glm_beta']):
        raise ValueError('Duplicate hypotheses in the direction references')
    result = {'glm_beta': {**who['glm_beta'], **event['glm_beta']},
              'p1_roles': who['p1_roles'], 'p1_profile': who['p1_profile']}
    (ROOT / args.out).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
