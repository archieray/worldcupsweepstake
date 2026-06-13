#!/usr/bin/env python3
"""
Sweepstake standings — reads wc2026_predictions.csv + sweepstake.json
and shows each player's teams with win probabilities and expected points.

Scoring (per team, per stage reached):
  Group exit → 0 pts   R32 → 1 pt    R16 → 3 pts
  QF → 6 pts           SF → 10 pts   Final → 15 pts   Winner → 25 pts

Usage:
  python3 standings.py            # predicted standings (pre-tournament)
  python3 standings.py --live     # live standings using match_results.json
"""

import csv
import json
import os
import sys

DATA_DIR = '/Users/archie-raythompson/Sweepstake'

STAGE_POINTS = {
    'Group':  0,
    'R32':    1,
    'R16':    3,
    'QF':     6,
    'SF':     10,
    '3rd':    10,   # same as SF (lost in semis)
    'Final':  15,
    'Winner': 25,
}

STAGE_ORDER = ['Group', 'R32', 'R16', 'QF', 'SF', '3rd', 'Final', 'Winner']

def load_sweepstake():
    with open(f'{DATA_DIR}/sweepstake.json') as f:
        return json.load(f)['players']

def load_predictions():
    probs = {}
    with open(f'{DATA_DIR}/wc2026_predictions.csv') as f:
        for row in csv.DictReader(f):
            probs[row['Team']] = {
                'group': row['Group'],
                'elo':   float(row['ELO']),
                'Win%':      float(row['Win%']),
                'Final%':    float(row['Final%']),
                'SF%':       float(row['SF%']),
                'QF%':       float(row['QF%']),
                'R16%':      float(row['R16%']),
                'R32%':      float(row['R32%']),
                'GroupExit%':float(row['GroupExit%']),
            }
    return probs

def load_live_results():
    path = f'{DATA_DIR}/match_results.json'
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    # Build {team: stage_reached} from knockout results
    # This is a simplified live tracker — stages locked in as results arrive
    return data

def expected_points(probs):
    """Compute expected points from stage probability distribution."""
    p = probs
    # Marginal probabilities per stage (not cumulative)
    p_group  = p['GroupExit%'] / 100
    p_r32    = (p['R32%'] - p['R16%']) / 100
    p_r16    = (p['R16%'] - p['QF%']) / 100
    p_qf     = (p['QF%'] - p['SF%']) / 100
    p_sf     = (p['SF%'] - p['Final%']) / 100
    p_final  = (p['Final%'] - p['Win%']) / 100
    p_winner = p['Win%'] / 100

    return (
        p_group  * STAGE_POINTS['Group']  +
        p_r32    * STAGE_POINTS['R32']    +
        p_r16    * STAGE_POINTS['R16']    +
        p_qf     * STAGE_POINTS['QF']     +
        p_sf     * STAGE_POINTS['SF']     +
        p_final  * STAGE_POINTS['Final']  +
        p_winner * STAGE_POINTS['Winner']
    )

def print_standings(players, predictions):
    # Compute expected points per player
    player_data = []
    for player, teams in players.items():
        team_rows = []
        total_exp = 0.0
        for team in teams:
            if team not in predictions:
                print(f"Warning: '{team}' not found in predictions CSV.")
                continue
            p    = predictions[team]
            exp  = expected_points(p)
            total_exp += exp
            team_rows.append((team, p['group'], p['elo'], p['Win%'], exp))
        player_data.append((player, total_exp, team_rows))

    player_data.sort(key=lambda x: -x[1])

    print(f"\n{'='*90}")
    print(f"  2026 WORLD CUP SWEEPSTAKE STANDINGS  |  Based on {10000:,}-simulation model")
    print(f"{'='*90}")

    for rank, (player, total_exp, teams) in enumerate(player_data, 1):
        print(f"\n  #{rank}  {player:<10}  Expected pts: {total_exp:.2f}")
        print(f"  {'─'*70}")
        print(f"  {'Team':<28} {'Grp':>3}  {'ELO':>4}  {'Win%':>6}  {'Exp pts':>7}")
        for team, grp, elo, win_pct, exp in sorted(teams, key=lambda r: -r[4]):
            print(f"  {team:<28} {grp:>3}  {elo:>4.0f}  {win_pct:>5.1f}%  {exp:>7.2f}")

    print(f"\n{'='*90}")
    print("Scoring: R32=1, R16=3, QF=6, SF=10, Final=15, Winner=25\n")


if __name__ == '__main__':
    players    = load_sweepstake()
    predictions = load_predictions()
    print_standings(players, predictions)
