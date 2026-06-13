#!/usr/bin/env python3
"""
Flask web frontend for the WC 2026 sweepstake predictor.
Run with: python3 app.py
Then open http://localhost:5000
"""

import csv
import json
import math
import os
import time
from datetime import datetime, timezone

import requests
from flask import Flask, render_template

app = Flask(__name__)

DATA_DIR = '/Users/archie-raythompson/Sweepstake'
API_KEY  = os.environ.get('FOOTBALL_DATA_API_KEY', '')
API_BASE = 'https://api.football-data.org/v4'
API_TTL  = 300  # seconds

_cache = {}
_throttle_until = 0.0  # epoch seconds — don't call API before this time

STAGE_POINTS = {'Group': 0, 'R32': 1, 'R16': 3, 'QF': 6, 'SF': 10, '3rd': 10, 'Final': 15, 'Winner': 25}

# Maps football-data.org team names → our canonical names
FD_NAME_MAP = {
    'Korea Republic':       'South Korea',
    "Côte d'Ivoire":        'Ivory Coast',
    'IR Iran':              'Iran',
    'Bosnia-Herzegovina':   'Bosnia and Herzegovina',
    'Congo DR':             'DR Congo',
    'Turkey':               'Türkiye',
    'Cabo Verde':           'Cape Verde',
    'USA':                  'United States',
    'Curaçao':              'Curaçao',
}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_predictions():
    probs = {}
    with open(f'{DATA_DIR}/wc2026_predictions.csv') as f:
        for row in csv.DictReader(f):
            probs[row['Team']] = {
                'group':      row['Group'],
                'elo':        float(row['ELO']),
                'win':        float(row['Win%']),
                'final':      float(row['Final%']),
                'sf':         float(row['SF%']),
                'qf':         float(row['QF%']),
                'r16':        float(row['R16%']),
                'r32':        float(row['R32%']),
                'group_exit': float(row['GroupExit%']),
            }
    return probs


def load_sweepstake():
    with open(f'{DATA_DIR}/sweepstake.json') as f:
        return json.load(f)['players']


def team_to_player(players):
    return {team: player for player, teams in players.items() for team in teams}


def expected_points(p):
    p_group  = p['group_exit'] / 100
    p_r32    = (p['r32']   - p['r16'])   / 100
    p_r16    = (p['r16']   - p['qf'])    / 100
    p_qf     = (p['qf']    - p['sf'])    / 100
    p_sf     = (p['sf']    - p['final']) / 100
    p_final  = (p['final'] - p['win'])   / 100
    p_winner = p['win'] / 100
    return (p_r32 * STAGE_POINTS['R32'] + p_r16 * STAGE_POINTS['R16'] +
            p_qf  * STAGE_POINTS['QF']  + p_sf  * STAGE_POINTS['SF']  +
            p_final * STAGE_POINTS['Final'] + p_winner * STAGE_POINTS['Winner'])


def compute_standings(players, predictions):
    rows = []
    for player, teams in players.items():
        team_rows = []
        total = 0.0
        for team in teams:
            p = predictions.get(team, {})
            if not p:
                continue
            exp = expected_points(p)
            total += exp
            team_rows.append({
                'name':  team,
                'group': p['group'],
                'elo':   int(p['elo']),
                'win':   p['win'],
                'exp':   round(exp, 2),
            })
        team_rows.sort(key=lambda r: -r['exp'])
        rows.append({'player': player, 'total': round(total, 2), 'teams': team_rows})
    rows.sort(key=lambda r: -r['total'])
    return rows


# ── API helpers ───────────────────────────────────────────────────────────────

def fetch_api(path):
    global _throttle_until
    now = time.time()

    # Serve from cache if still fresh
    if path in _cache:
        data, ts = _cache[path]
        if now - ts < API_TTL:
            return data

    if not API_KEY:
        return None

    # Respect any throttle window set by a previous response
    if now < _throttle_until:
        # Return stale cache rather than hammering the API
        if path in _cache:
            return _cache[path][0]
        return None

    try:
        r = requests.get(f'{API_BASE}{path}', headers={'X-Auth-Token': API_KEY}, timeout=8)

        # football-data.org rate-limit headers:
        #   X-Requests-Available-Minute  — remaining calls this minute
        #   X-RequestCounter-Reset       — seconds until the window resets
        available = r.headers.get('X-Requests-Available-Minute')
        reset_in  = r.headers.get('X-RequestCounter-Reset')

        if available is not None and int(available) <= 1:
            # Almost out of quota — hold off until the window resets
            wait = int(reset_in) if reset_in else 60
            _throttle_until = time.time() + wait

        if r.status_code == 429:
            # Hard rate-limit hit — back off for the reset window
            wait = int(reset_in) if reset_in else 60
            _throttle_until = time.time() + wait
            if path in _cache:
                return _cache[path][0]
            return None

        r.raise_for_status()
        data = r.json()
        _cache[path] = (data, time.time())
        return data

    except Exception:
        return None


def canonical(name):
    return FD_NAME_MAP.get(name, name)


def load_regressions():
    """Load fitted Poisson regression coefficients from the model cache."""
    path = f'{DATA_DIR}/wc2026_cache.json'
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f).get('regressions', {})


def poisson_match_probs(team_a, team_b, elo_a, elo_b, regressions, max_goals=8):
    """
    Compute win/draw/loss probabilities for a neutral-venue match using the
    fitted per-team Poisson regressions (Gilch 2022).
    Returns (p_a_win, p_draw, p_b_win) as percentages rounded to integers.
    """
    reg = regressions
    if team_a not in reg or team_b not in reg:
        # Fallback: ELO-based estimate, no draw
        diff = elo_a - elo_b
        p_a = 1 / (1 + 10 ** (-diff / 400))
        p_a_pct = round(p_a * 100)
        return p_a_pct, 0, 100 - p_a_pct

    def lam(scorer, conceder, elo_opp):
        sc = reg[scorer]['scored']
        co = reg[conceder]['conceded']
        mu = math.exp(sc[0] + sc[1] * elo_opp)       # location=0 (neutral)
        nu = math.exp(co[0] + co[1] * elo_opp)
        return (mu + nu) / 2

    la = lam(team_a, team_b, elo_b)
    lb = lam(team_b, team_a, elo_a)

    def poisson_pmf(lam, k):
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    p_win = p_draw = p_loss = 0.0
    for i in range(max_goals + 1):
        pi = poisson_pmf(la, i)
        for j in range(max_goals + 1):
            pj = poisson_pmf(lb, j)
            p = pi * pj
            if i > j:
                p_win += p
            elif i == j:
                p_draw += p
            else:
                p_loss += p

    total = p_win + p_draw + p_loss
    p_win  = round(p_win  / total * 100)
    p_draw = round(p_draw / total * 100)
    p_loss = 100 - p_win - p_draw
    return p_win, p_draw, p_loss


def get_scorers():
    data = fetch_api('/competitions/WC/scorers?limit=20')
    if not data:
        return {}
    by_team = {}
    for s in data.get('scorers', []):
        team = canonical(s['team']['name'])
        by_team.setdefault(team, []).append({
            'name':  s['player']['name'],
            'goals': s.get('goals') or s.get('numberOfGoals', 0),
        })
    return by_team


def get_team_goals():
    """Returns {team: goals_scored_this_wc} from match results."""
    goals = {}
    path = f'{DATA_DIR}/match_results.json'
    if not os.path.exists(path):
        return goals
    with open(path) as f:
        results = json.load(f)
    for key, m in results.get('group', {}).items():
        h, a = m['home'], m['away']
        goals[h] = goals.get(h, 0) + m['home_score']
        goals[a] = goals.get(a, 0) + m['away_score']
    for slot, m in results.get('knockout', {}).items():
        h, a = m['team_a'], m['team_b']
        goals[h] = goals.get(h, 0) + m['score_a']
        goals[a] = goals.get(a, 0) + m['score_b']
    return goals


def get_fixtures():
    data = fetch_api('/competitions/WC/matches?status=SCHEDULED')
    if not data:
        return []
    matches = data.get('matches', [])
    matches.sort(key=lambda m: m['utcDate'])
    out = []
    for m in matches[:10]:
        home = canonical(m['homeTeam']['name'])
        away = canonical(m['awayTeam']['name'])
        dt   = datetime.fromisoformat(m['utcDate'].replace('Z', '+00:00'))
        out.append({
            'home':    home,
            'away':    away,
            'date':    dt.strftime('%a %d %b'),
            'time':    dt.strftime('%H:%M UTC'),
            'stage':   m.get('stage', ''),
            'matchday': m.get('matchday', ''),
        })
    return out


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def standings():
    players     = load_sweepstake()
    predictions = load_predictions()
    rows        = compute_standings(players, predictions)
    return render_template('standings.html', rows=rows, active='standings')


@app.route('/teams')
def teams():
    predictions = load_predictions()
    players     = load_sweepstake()
    t2p         = team_to_player(players)
    scorers     = get_scorers()
    goals       = get_team_goals()

    # Group teams by group letter
    grouped = {}
    for team, p in predictions.items():
        g = p['group']
        grouped.setdefault(g, []).append({
            'name':    team,
            'elo':     int(p['elo']),
            'win':     p['win'],
            'r16':     p['r16'],
            'qf':      p['qf'],
            'goals':   goals.get(team, 0),
            'scorers': scorers.get(team, [])[:3],
            'owner':   t2p.get(team, '—'),
            'exp':     round(expected_points(p), 2),
        })

    for g in grouped:
        grouped[g].sort(key=lambda t: -t['elo'])

    groups = sorted(grouped.items())
    has_api = bool(API_KEY)
    return render_template('teams.html', groups=groups, active='teams', has_api=has_api)


@app.route('/fixtures')
def fixtures():
    predictions  = load_predictions()
    players      = load_sweepstake()
    regressions  = load_regressions()
    t2p          = team_to_player(players)
    raw          = get_fixtures()
    has_api      = bool(API_KEY)

    enriched = []
    for m in raw:
        h, a = m['home'], m['away']
        ph = predictions.get(h, {})
        pa = predictions.get(a, {})
        elo_h = ph.get('elo', 1500)
        elo_a = pa.get('elo', 1500)

        h_pct, draw_pct, a_pct = poisson_match_probs(h, a, elo_h, elo_a, regressions)

        enriched.append({
            **m,
            'home_owner': t2p.get(h, '—'),
            'away_owner': t2p.get(a, '—'),
            'home_pct':   h_pct,
            'draw_pct':   draw_pct,
            'away_pct':   a_pct,
            'home_elo':   int(elo_h),
            'away_elo':   int(elo_a),
        })

    return render_template('fixtures.html', fixtures=enriched, active='fixtures', has_api=has_api)


if __name__ == '__main__':
    if not API_KEY:
        print("Note: FOOTBALL_DATA_API_KEY not set — live stats and fixtures will be unavailable.")
        print("      Get a free key at https://www.football-data.org/client/register")
        print("      Then: export FOOTBALL_DATA_API_KEY=your_key_here")
    print("\nStarting server at http://localhost:5001\n")
    app.run(debug=True, port=5001)
