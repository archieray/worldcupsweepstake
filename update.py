#!/usr/bin/env python3
"""
Record a match result and re-run the WC 2026 simulation.

Usage:
  Group stage:
    python update.py group "Mexico" 2 "South Korea" 1

  Knockout match (winner from score):
    python update.py knockout R32-1 "South Korea" 0 "Canada" 2

  Knockout match that was drawn (requires --winner):
    python update.py knockout R32-1 "South Korea" 1 "Canada" 1 --winner "Canada"

  Remove a result:
    python update.py remove group "Mexico" "South Korea"
    python update.py remove knockout R32-1
"""

import argparse
import json
import os
import sys

DATA_DIR         = '/Users/archie-raythompson/Sweepstake'
RESULTS_PATH     = f'{DATA_DIR}/match_results.json'

# Valid knockout slot IDs
VALID_KO_SLOTS = (
    [f'R32-{i}' for i in range(1, 17)] +
    [f'R16-{i}' for i in range(1, 9)] +
    [f'QF{i}'   for i in range(1, 5)] +
    ['SF1', 'SF2', 'Final']
)

# Import group/team data from predictor
sys.path.insert(0, DATA_DIR)
from wc2026_predictor import GROUPS

# Build lookup helpers
TEAM_TO_GROUP = {t: g for g, members in GROUPS.items() for t in members}
ALL_TEAMS     = set(TEAM_TO_GROUP)


def load_results():
    if not os.path.exists(RESULTS_PATH):
        return {'group': {}, 'knockout': {}}
    with open(RESULTS_PATH) as f:
        return json.load(f)


def save_results(data):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def group_key(team_a, team_b):
    return '|'.join(sorted([team_a, team_b]))


def validate_team(name):
    if name not in ALL_TEAMS:
        close = [t for t in ALL_TEAMS if name.lower() in t.lower()]
        hint  = f'  Did you mean: {", ".join(close)}' if close else ''
        sys.exit(f'Error: unknown team "{name}".{hint}')


def cmd_group(args):
    team_a, score_a, team_b, score_b = args.team_a, args.score_a, args.team_b, args.score_b
    validate_team(team_a)
    validate_team(team_b)
    if TEAM_TO_GROUP[team_a] != TEAM_TO_GROUP[team_b]:
        sys.exit(f'Error: {team_a} (Group {TEAM_TO_GROUP[team_a]}) and '
                 f'{team_b} (Group {TEAM_TO_GROUP[team_b]}) are not in the same group.')

    data = load_results()
    key  = group_key(team_a, team_b)
    data['group'][key] = {
        'home': team_a, 'away': team_b,
        'home_score': score_a, 'away_score': score_b,
    }
    save_results(data)
    print(f'Recorded: {team_a} {score_a}–{score_b} {team_b}  '
          f'(Group {TEAM_TO_GROUP[team_a]})')


def cmd_knockout(args):
    slot, team_a, score_a, team_b, score_b = (
        args.slot, args.team_a, args.score_a, args.team_b, args.score_b)
    validate_team(team_a)
    validate_team(team_b)

    if slot not in VALID_KO_SLOTS:
        sys.exit(f'Error: "{slot}" is not a valid slot ID.\n'
                 f'Valid IDs: {", ".join(VALID_KO_SLOTS)}')

    if score_a == score_b:
        if not args.winner:
            sys.exit('Error: scores are level — knockout matches need a winner.\n'
                     f'Add: --winner "{team_a}"  or  --winner "{team_b}"')
        if args.winner not in (team_a, team_b):
            sys.exit(f'Error: --winner must be "{team_a}" or "{team_b}".')
        winner = args.winner
    else:
        winner = team_a if score_a > score_b else team_b

    data = load_results()
    data['knockout'][slot] = {
        'team_a': team_a, 'team_b': team_b,
        'score_a': score_a, 'score_b': score_b,
        'winner': winner,
    }
    save_results(data)
    print(f'Recorded: [{slot}] {team_a} {score_a}–{score_b} {team_b}  →  Winner: {winner}')


def cmd_remove(args):
    data = load_results()
    if args.stage == 'group':
        validate_team(args.ref_a)
        validate_team(args.ref_b)
        key = group_key(args.ref_a, args.ref_b)
        if key not in data['group']:
            sys.exit(f'No recorded result for {args.ref_a} vs {args.ref_b}.')
        del data['group'][key]
        print(f'Removed group result: {args.ref_a} vs {args.ref_b}')
    elif args.stage == 'knockout':
        slot = args.ref_a
        if slot not in data['knockout']:
            sys.exit(f'No recorded result for slot {slot}.')
        del data['knockout'][slot]
        print(f'Removed knockout result: {slot}')
    save_results(data)


def rerun():
    print('\nRe-running simulation...')
    from wc2026_predictor import main as predictor_main
    predictor_main()


def build_parser():
    p = argparse.ArgumentParser(description='Record WC 2026 results and re-simulate.')
    sub = p.add_subparsers(dest='command', required=True)

    # group subcommand
    g = sub.add_parser('group', help='Record a group-stage result')
    g.add_argument('team_a', metavar='TEAM_A')
    g.add_argument('score_a', metavar='SCORE_A', type=int)
    g.add_argument('team_b', metavar='TEAM_B')
    g.add_argument('score_b', metavar='SCORE_B', type=int)

    # knockout subcommand
    k = sub.add_parser('knockout', help='Record a knockout-round result')
    k.add_argument('slot',    metavar='SLOT',    help='e.g. R32-1, R16-3, QF2, SF1, Final')
    k.add_argument('team_a',  metavar='TEAM_A')
    k.add_argument('score_a', metavar='SCORE_A', type=int)
    k.add_argument('team_b',  metavar='TEAM_B')
    k.add_argument('score_b', metavar='SCORE_B', type=int)
    k.add_argument('--winner', metavar='TEAM', default=None,
                   help='Required when scores are level (AET/penalties)')

    # remove subcommand
    r = sub.add_parser('remove', help='Remove a previously recorded result')
    r.add_argument('stage', choices=['group', 'knockout'])
    r.add_argument('ref_a', metavar='TEAM_A_or_SLOT',
                   help='For group: first team name. For knockout: slot ID (e.g. R32-1)')
    r.add_argument('ref_b', metavar='TEAM_B', nargs='?', default=None,
                   help='For group: second team name (not needed for knockout)')

    return p


if __name__ == '__main__':
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == 'group':
        cmd_group(args)
    elif args.command == 'knockout':
        cmd_knockout(args)
    elif args.command == 'remove':
        cmd_remove(args)

    rerun()
