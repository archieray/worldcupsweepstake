#!/usr/bin/env python3
"""
World Cup 2026 Poisson Prediction Model
----------------------------------------
Per-team Independent Poisson regression fitted on all international matches
since 2016, weighted by recency and match importance. Opponent ELO (computed
from full match history) and match location are the covariates, following the
approach of Gilch (2022). Full fixed-bracket Monte Carlo simulation of the
2026 tournament including correct R32 seeding for best 8 third-place teams.
"""

import csv
import math
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
from itertools import combinations
from datetime import date

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR       = '/Users/archie-raythompson/Sweepstake'
N_SIMS         = 10_000
RANDOM_SEED    = 42
CUTOFF_DATE    = '2016-01-01'
HALF_LIFE_DAYS = 365 * 3
TODAY          = date.today().isoformat()
ELO_START      = 1500
ELO_K_BASE     = 15

# ── Name mapping: canonical → results.csv ────────────────────────────────────
# Canonical names are those used in GROUPS below.
RESULTS_NAME = {
    'Bosnia and Herzegovina': 'Bosnia and Herzegovina',  # already matches
    'DR Congo':               'DR Congo',                # already matches
    'Ivory Coast':            'Ivory Coast',             # already matches
    'South Korea':            'South Korea',             # already matches
    'Iran':                   'Iran',                    # already matches
    'Czechia':                'Czech Republic',
    'Türkiye':                'Turkey',
    'Curaçao':                'Curaçao',
    'United States':          'United States',
}

# ── Tournament importance weights ────────────────────────────────────────────
IMPORTANCE = {
    'FIFA World Cup':                          4,
    'UEFA Euro':                               3,
    'Copa América':                            3,
    'African Cup of Nations':                  3,
    'AFC Asian Cup':                           3,
    'Gold Cup':                                3,
    'Oceania Nations Cup':                     3,
    'Confederations Cup':                      3,
    'CONMEBOL–UEFA Cup of Champions':          3,
    'FIFA World Cup qualification':            2.5,
    'UEFA Euro qualification':                 2.5,
    'UEFA Nations League':                     2.5,
    'CONCACAF Nations League':                 2.5,
    'CONCACAF Nations League qualification':   2.5,
    'Copa América qualification':              2.5,
    'AFC Asian Cup qualification':             2.5,
    'African Cup of Nations qualification':    2.5,
    'FIFA Series':                             2.5,
}

# ── Tournament structure ──────────────────────────────────────────────────────
GROUPS = {
    'A': ['Mexico',    'South Korea',           'South Africa', 'Czechia'],
    'B': ['Canada',    'Bosnia and Herzegovina', 'Qatar',       'Switzerland'],
    'C': ['Brazil',    'Morocco',               'Haiti',        'Scotland'],
    'D': ['United States', 'Paraguay',          'Australia',    'Türkiye'],
    'E': ['Germany',   'Curaçao',               'Ivory Coast',  'Ecuador'],
    'F': ['Netherlands','Japan',                'Sweden',       'Tunisia'],
    'G': ['Belgium',   'Egypt',                 'Iran',         'New Zealand'],
    'H': ['Spain',     'Cape Verde',            'Saudi Arabia', 'Uruguay'],
    'I': ['France',    'Senegal',               'Iraq',         'Norway'],
    'J': ['Argentina', 'Algeria',               'Austria',      'Jordan'],
    'K': ['Portugal',  'DR Congo',              'Uzbekistan',   'Colombia'],
    'L': ['England',   'Croatia',               'Ghana',        'Panama'],
}

# R32 slot definitions: (slot_id, team_a_label, team_b_label)
# Labels: 'XN' = Nth place in Group X; None = filled by best-3rd assignment
R32_FIXED = [
    ('R32-1',  'A2', 'B2'),
    ('R32-2',  'E1', None),
    ('R32-3',  'F1', 'C2'),
    ('R32-4',  'C1', 'F2'),
    ('R32-5',  'I1', None),
    ('R32-6',  'E2', 'I2'),
    ('R32-7',  'A1', None),
    ('R32-8',  'L1', None),
    ('R32-9',  'D1', None),
    ('R32-10', 'G1', None),
    ('R32-11', 'K2', 'L2'),
    ('R32-12', 'J1', 'H2'),
    ('R32-13', 'H1', 'J2'),
    ('R32-14', 'B1', None),
    ('R32-15', 'D2', 'G2'),
    ('R32-16', 'K1', None),
]

# Which groups each third-place slot may draw from
THIRD_PLACE_SLOTS = {
    'R32-2':  ['A','B','C','D','F'],
    'R32-5':  ['C','D','F','G','H'],
    'R32-7':  ['C','E','F','H','I'],
    'R32-8':  ['E','H','I','J','K'],
    'R32-9':  ['B','E','F','I','J'],
    'R32-10': ['A','E','H','I','J'],
    'R32-14': ['A','D','G','H','K'],
    'R32-16': ['B','C','G','J','L'],
}

# Fixed knockout pairings (each entry is two slot labels whose winners meet)
R16_PAIRS = [
    ('R32-1','R32-2'), ('R32-3','R32-4'), ('R32-5','R32-6'),  ('R32-7','R32-8'),
    ('R32-9','R32-10'),('R32-11','R32-12'),('R32-13','R32-14'),('R32-15','R32-16'),
]
QF_PAIRS = [
    ('R16-1','R16-2'), ('R16-5','R16-6'),
    ('R16-3','R16-4'), ('R16-7','R16-8'),
]
SF_PAIRS = [('QF1','QF2'), ('QF3','QF4')]

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    with open(f'{DATA_DIR}/results.csv') as f:
        results = list(csv.DictReader(f))
    with open(f'{DATA_DIR}/teams.csv') as f:
        current_elo = {r['team']: float(r['elo']) for r in csv.DictReader(f)}
    return results, current_elo

# ── Historical ELO computation ────────────────────────────────────────────────

def _elo_G(gd):
    if gd <= 1:   return 1.0
    elif gd == 2: return 1.5
    else:         return (11 + gd) / 8

def _elo_We(elo_self, elo_opp):
    return 1.0 / (10 ** (-(elo_self - elo_opp) / 400) + 1)

def compute_historical_elo(results):
    elo = defaultdict(lambda: ELO_START)
    elo_history = {}

    for r in sorted(results, key=lambda r: r['date']):
        h, a = r['home_team'], r['away_team']
        try:
            hg, ag = int(r['home_score']), int(r['away_score'])
        except (ValueError, TypeError):
            continue

        K = ELO_K_BASE * IMPORTANCE.get(r['tournament'], 1)
        elo_h, elo_a = elo[h], elo[a]
        elo_history[(r['date'], h, a)] = (elo_h, elo_a)

        gd = abs(hg - ag)
        G  = _elo_G(gd)
        W_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        W_a = 1.0 - W_h if hg != ag else 0.5

        elo[h] = elo_h + K * G * (W_h - _elo_We(elo_h, elo_a))
        elo[a] = elo_a + K * G * (W_a - _elo_We(elo_a, elo_h))

    return elo_history, dict(elo)

# ── Per-team training dataset ─────────────────────────────────────────────────

def _recency_weight(match_date_str):
    days_ago = (date.fromisoformat(TODAY) - date.fromisoformat(match_date_str)).days
    return 0.5 ** (days_ago / HALF_LIFE_DAYS)

def build_team_datasets(results, elo_history, sched_teams):
    # Build reverse map: results.csv name → canonical name
    rev = {}
    for canon, res_name in RESULTS_NAME.items():
        rev[res_name] = canon
    # Also add identity mappings for names that match directly
    for t in sched_teams:
        if t not in rev.values():
            rev[t] = t

    team_data = {t: [] for t in sched_teams}

    for r in results:
        if r['date'] < CUTOFF_DATE:
            continue
        try:
            hg, ag = int(r['home_score']), int(r['away_score'])
        except (ValueError, TypeError):
            continue

        h_res, a_res = r['home_team'], r['away_team']
        h = rev.get(h_res, h_res)
        a = rev.get(a_res, a_res)

        key = (r['date'], r['home_team'], r['away_team'])
        if key not in elo_history:
            continue
        elo_h_before, elo_a_before = elo_history[key]

        imp = IMPORTANCE.get(r['tournament'], 1)
        w   = _recency_weight(r['date']) * imp
        neutral = r.get('neutral', 'FALSE').strip().upper() == 'TRUE'
        loc_h =  0 if neutral else  1
        loc_a =  0 if neutral else -1

        if h in team_data:
            team_data[h].append({'goals_scored': hg, 'goals_conceded': ag,
                                  'elo_opp': elo_a_before, 'location': loc_h, 'weight': w})
        if a in team_data:
            team_data[a].append({'goals_scored': ag, 'goals_conceded': hg,
                                  'elo_opp': elo_h_before, 'location': loc_a, 'weight': w})

    return team_data

# ── Weighted Poisson regression ───────────────────────────────────────────────

def _fit_poisson(goals, elo_opps, locations, weights):
    goals     = np.array(goals,     dtype=float)
    elo_opps  = np.array(elo_opps,  dtype=float)
    locations = np.array(locations, dtype=float)
    weights   = np.array(weights,   dtype=float)

    elo_mean = elo_opps.mean()
    elo_std  = elo_opps.std() + 1e-8
    elo_norm = (elo_opps - elo_mean) / elo_std

    def neg_ll(params):
        a0, a1, a2 = params
        mu = np.exp(np.clip(a0 + a1 * elo_norm + a2 * locations, -10, 10))
        return -(weights * (goals * np.log(mu + 1e-12) - mu)).sum()

    res = minimize(neg_ll, [np.log(1.3), 0.0, 0.0], method='L-BFGS-B',
                   options={'maxiter': 1000, 'ftol': 1e-10})
    a0, a1, a2 = res.x
    return [a0 - a1 * elo_mean / elo_std, a1 / elo_std, a2]

def fit_team_regressions(team_data):
    MIN_MATCHES = 5

    # Pooled fallback
    pg, pe, pl, pw = [], [], [], []
    cg, ce, cl, cw = [], [], [], []
    for records in team_data.values():
        for r in records:
            pg.append(r['goals_scored']);   pe.append(r['elo_opp'])
            pl.append(r['location']);       pw.append(r['weight'])
            cg.append(r['goals_conceded']); ce.append(r['elo_opp'])
            cl.append(r['location']);       cw.append(r['weight'])
    pooled_s = _fit_poisson(pg, pe, pl, pw)
    pooled_c = _fit_poisson(cg, ce, cl, cw)

    regressions = {}
    for team, records in team_data.items():
        if len(records) < MIN_MATCHES:
            regressions[team] = {'scored': pooled_s, 'conceded': pooled_c}
            continue
        gs = [r['goals_scored']   for r in records]
        gc = [r['goals_conceded'] for r in records]
        elos = [r['elo_opp']      for r in records]
        locs = [r['location']     for r in records]
        ws   = [r['weight']       for r in records]
        regressions[team] = {
            'scored':   _fit_poisson(gs, elos, locs, ws),
            'conceded': _fit_poisson(gc, elos, locs, ws),
        }
    return regressions

# ── Expected goals ────────────────────────────────────────────────────────────

def expected_goals(team_a, team_b, elo_a, elo_b, regressions, loc_a=0, loc_b=0):
    def pred(coef, elo_opp, loc):
        return math.exp(coef[0] + coef[1] * elo_opp + coef[2] * loc)

    reg_a, reg_b = regressions[team_a], regressions[team_b]
    mu_a = pred(reg_a['scored'],   elo_b, loc_a)
    nu_b = pred(reg_b['conceded'], elo_a, loc_b)
    mu_b = pred(reg_b['scored'],   elo_a, loc_b)
    nu_a = pred(reg_a['conceded'], elo_b, loc_a)
    return (mu_a + nu_b) / 2, (mu_b + nu_a) / 2

# ── Match simulation ──────────────────────────────────────────────────────────

def sim_match(lam_h, lam_a, rng):
    return rng.poisson(lam_h), rng.poisson(lam_a)

def sim_knockout_match(lam_h, lam_a, rng):
    g1, g2 = sim_match(lam_h, lam_a, rng)
    if g1 != g2:
        return 0 if g1 > g2 else 1
    p1 = lam_h / (lam_h + lam_a + 1e-10)
    return 0 if rng.random() < p1 else 1

# ── Group stage ───────────────────────────────────────────────────────────────

def sim_group_stage(strengths, current_elo, rng):
    """
    Simulate all 12 groups. Returns:
        standings  : {team: {pts, gf, ga, gd, group, group_rank}}
        group_tables: {'A': [1st, 2nd, 3rd, 4th], ...}
    """
    standings = {}
    for letter, members in GROUPS.items():
        for team in members:
            standings[team] = {'pts': 0, 'gf': 0, 'ga': 0, 'gd': 0,
                               'group': letter}

    group_tables = {}
    for letter, members in GROUPS.items():
        for h, a in combinations(members, 2):
            lam_h, lam_a = strengths(h, a)
            gh, ga = sim_match(lam_h, lam_a, rng)

            standings[h]['gf'] += gh; standings[h]['ga'] += ga
            standings[a]['gf'] += ga; standings[a]['ga'] += gh
            if gh > ga:
                standings[h]['pts'] += 3
            elif gh == ga:
                standings[h]['pts'] += 1; standings[a]['pts'] += 1
            else:
                standings[a]['pts'] += 3

        for t in members:
            standings[t]['gd'] = standings[t]['gf'] - standings[t]['ga']

        ranked = sorted(
            members,
            key=lambda t: (
                standings[t]['pts'],
                standings[t]['gd'],
                standings[t]['gf'],
                rng.random(),                          # discipline (randomised)
                current_elo.get(t, ELO_START),         # ELO proxy for FIFA rank
            ),
            reverse=True,
        )
        for rank, team in enumerate(ranked):
            standings[team]['group_rank'] = rank + 1
        group_tables[letter] = ranked

    return standings, group_tables

# ── Third-place assignment ────────────────────────────────────────────────────

def pick_best_third(standings, group_tables, current_elo, rng):
    """Return all 12 third-place teams ranked best→worst."""
    thirds = [ranked[2] for ranked in group_tables.values()]
    return sorted(
        thirds,
        key=lambda t: (
            standings[t]['pts'],
            standings[t]['gd'],
            standings[t]['gf'],
            rng.random(),
            current_elo.get(t, ELO_START),
        ),
        reverse=True,
    )

def assign_third_place_teams(ranked_thirds, standings):
    """
    Assign the best 8 third-place teams to the 8 R32 slots that require one.
    Uses greedy assignment ordered by most-constrained slot first.
    Returns {slot_id: team}.
    """
    best8 = ranked_thirds[:8]
    group_of = {t: standings[t]['group'] for t in best8}

    # Sort slots by fewest eligible groups (most constrained first)
    slots_by_constraint = sorted(
        THIRD_PLACE_SLOTS.items(),
        key=lambda kv: len(kv[1]),
    )

    assignment = {}
    assigned_teams = set()

    for slot_id, eligible_groups in slots_by_constraint:
        for team in best8:
            if team not in assigned_teams and group_of[team] in eligible_groups:
                assignment[slot_id] = team
                assigned_teams.add(team)
                break
        else:
            # Fallback: take any unassigned team (shouldn't happen with valid bracket)
            for team in best8:
                if team not in assigned_teams:
                    assignment[slot_id] = team
                    assigned_teams.add(team)
                    break

    return assignment

# ── R32 bracket resolution ────────────────────────────────────────────────────

def resolve_r32_bracket(group_tables, third_assignment):
    """
    Build the 16 R32 matchups as [(slot_id, team_a, team_b), ...].
    Labels like 'A1' = winner of Group A, 'A2' = runner-up, etc.
    """
    def get_team(label):
        if label is None:
            return None
        group = label[0]
        rank  = int(label[1]) - 1
        return group_tables[group][rank]

    matchups = []
    for slot_id, label_a, label_b in R32_FIXED:
        team_a = get_team(label_a)
        team_b = get_team(label_b) if label_b is not None else third_assignment[slot_id]
        matchups.append((slot_id, team_a, team_b))
    return matchups

# ── Generic knockout round ────────────────────────────────────────────────────

def sim_knockout_round(matchups, strengths, rng):
    """
    matchups: [(slot_id, team_a, team_b), ...]
    Returns: {slot_id: winning_team}
    """
    winners = {}
    for slot_id, t1, t2 in matchups:
        lam1, lam2 = strengths(t1, t2)
        w = sim_knockout_match(lam1, lam2, rng)
        winners[slot_id] = t1 if w == 0 else t2
    return winners

def build_next_round(pair_list, prev_winners, round_prefix):
    """
    Build matchups for the next knockout round.
    pair_list: [('R32-1','R32-2'), ...]  or  [('R16-1','R16-2'), ...]
    prev_winners: {slot_id: team}
    round_prefix: 'R16', 'QF', 'SF', etc.
    """
    matchups = []
    for i, (s1, s2) in enumerate(pair_list, 1):
        slot_id = f'{round_prefix}{i}'
        matchups.append((slot_id, prev_winners[s1], prev_winners[s2]))
    return matchups

# ── Full tournament simulation ────────────────────────────────────────────────

STAGE_ORDER = ['Group', 'R32', 'R16', 'QF', 'SF', '3rd', 'Final', 'Winner']

def sim_tournament(strengths, current_elo, rng):
    all_teams = [t for members in GROUPS.values() for t in members]
    result    = {t: 'Group' for t in all_teams}

    # ── Group stage ───────────────────────────────────────────────────────────
    standings, group_tables = sim_group_stage(strengths, current_elo, rng)
    ranked_thirds = pick_best_third(standings, group_tables, current_elo, rng)
    third_assignment = assign_third_place_teams(ranked_thirds, standings)

    # All group qualifiers
    qualifiers = set()
    for ranked in group_tables.values():
        qualifiers.add(ranked[0]); qualifiers.add(ranked[1])
    qualifiers |= set(third_assignment.values())
    for t in qualifiers:
        result[t] = 'R32'

    # ── Round of 32 ──────────────────────────────────────────────────────────
    r32_matchups = resolve_r32_bracket(group_tables, third_assignment)
    r32_winners  = sim_knockout_round(r32_matchups, strengths, rng)
    for t in r32_winners.values():
        result[t] = 'R16'

    # ── Round of 16 ──────────────────────────────────────────────────────────
    r16_matchups = build_next_round(R16_PAIRS, r32_winners, 'R16-')
    r16_winners  = sim_knockout_round(r16_matchups, strengths, rng)
    for t in r16_winners.values():
        result[t] = 'QF'

    # ── Quarter-finals ────────────────────────────────────────────────────────
    qf_matchups = build_next_round(QF_PAIRS, r16_winners, 'QF')
    qf_winners  = sim_knockout_round(qf_matchups, strengths, rng)
    for t in qf_winners.values():
        result[t] = 'SF'

    # ── Semi-finals ───────────────────────────────────────────────────────────
    sf_matchups = build_next_round(SF_PAIRS, qf_winners, 'SF')
    sf_winners  = sim_knockout_round(sf_matchups, strengths, rng)
    sf_losers   = [t2 if w == t1 else t1
                   for (_, t1, t2), w in zip(sf_matchups, sf_winners.values())]
    for t in sf_losers:
        result[t] = '3rd'
    for t in sf_winners.values():
        result[t] = 'Final'

    # ── Final ─────────────────────────────────────────────────────────────────
    final_matchup = [('Final', list(sf_winners.values())[0], list(sf_winners.values())[1])]
    final_winner  = sim_knockout_round(final_matchup, strengths, rng)
    result[final_winner['Final']] = 'Winner'

    return result

# ── Monte Carlo ───────────────────────────────────────────────────────────────

def run_simulations(strengths_fn, current_elo, n_sims=N_SIMS):
    rng       = np.random.default_rng(RANDOM_SEED)
    all_teams = [t for members in GROUPS.values() for t in members]
    counts    = {stage: defaultdict(int) for stage in STAGE_ORDER}

    for _ in range(n_sims):
        result = sim_tournament(strengths_fn, current_elo, rng)
        for team, stage in result.items():
            idx = STAGE_ORDER.index(stage)
            for s in STAGE_ORDER[:idx+1]:
                counts[s][team] += 1

    probs = {}
    for team in all_teams:
        probs[team] = {s: counts[s][team] / n_sims for s in STAGE_ORDER}
    return probs

# ── Output ────────────────────────────────────────────────────────────────────

def print_results(probs, regressions, current_elo):
    teams = sorted(probs, key=lambda t: -probs[t]['Winner'])

    print(f"\n{'='*105}")
    print(f"  2026 WORLD CUP PREDICTION  |  Per-team Poisson + ELO  |  {N_SIMS:,} simulations  |  Fixed bracket")
    print(f"{'='*105}")
    header = (f"{'Team':<26} {'Grp':>3}  {'ELO':>4}  "
              f"{'Win%':>6}  {'Final%':>7}  {'SF%':>6}  {'QF%':>6}  {'R16%':>6}  {'R32%':>6}  {'Groups%':>7}")
    print(header)
    print('-' * 105)

    group_of = {t: g for g, members in GROUPS.items() for t in members}
    for team in teams:
        p   = probs[team]
        elo = current_elo.get(team, 0)
        grp = group_of.get(team, '?')
        print(f"{team:<26} {grp:>3}  {elo:>4.0f}  "
              f"{p['Winner']*100:>5.1f}%  "
              f"{p['Final']*100:>6.1f}%  "
              f"{p['SF']*100:>5.1f}%  "
              f"{p['QF']*100:>5.1f}%  "
              f"{p['R16']*100:>5.1f}%  "
              f"{p['R32']*100:>5.1f}%  "
              f"{(1-p['R32'])*100:>6.1f}%")

    print('=' * 105)
    print("Note: 'Groups%' = probability of being eliminated in the group stage.\n")


def save_results_csv(probs, regressions, current_elo, filepath):
    group_of = {t: g for g, members in GROUPS.items() for t in members}
    teams    = sorted(probs, key=lambda t: -probs[t]['Winner'])
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Team','Group','ELO',
                         'Win%','Final%','SF%','QF%','R16%','R32%','GroupExit%'])
        for team in teams:
            p   = probs[team]
            writer.writerow([
                team, group_of.get(team,'?'), round(current_elo.get(team,0), 0),
                round(p['Winner']*100,2), round(p['Final']*100,2),
                round(p['SF']*100,2),    round(p['QF']*100,2),
                round(p['R16']*100,2),   round(p['R32']*100,2),
                round((1-p['R32'])*100,2),
            ])
    print(f"Results saved to {filepath}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    results, current_elo = load_data()

    print(f"Computing historical ELO from {len(results):,} matches...")
    elo_history, computed_elo = compute_historical_elo(results)

    sched_teams = [t for members in GROUPS.values() for t in members]
    print(f"Building per-team training datasets (since {CUTOFF_DATE})...")
    team_data = build_team_datasets(results, elo_history, sched_teams)
    counts    = {t: len(v) for t, v in team_data.items()}
    min_t = min(counts, key=counts.get)
    max_t = max(counts, key=counts.get)
    print(f"  Match counts: min={counts[min_t]} ({min_t}), max={counts[max_t]} ({max_t})")

    print("Fitting per-team Poisson regressions...")
    regressions = fit_team_regressions(team_data)

    def strengths(team_a, team_b):
        elo_a = current_elo.get(team_a, computed_elo.get(team_a, ELO_START))
        elo_b = current_elo.get(team_b, computed_elo.get(team_b, ELO_START))
        return expected_goals(team_a, team_b, elo_a, elo_b, regressions)

    print(f"Running {N_SIMS:,} tournament simulations...")
    probs = run_simulations(strengths, current_elo)

    print_results(probs, regressions, current_elo)
    save_results_csv(probs, regressions, current_elo,
                     f'{DATA_DIR}/wc2026_predictions.csv')


if __name__ == '__main__':
    main()
