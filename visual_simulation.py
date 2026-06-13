#!/usr/bin/env python3
"""
WC 2026 Visual Simulation
Runs one Monte Carlo simulation and generates an animated HTML bracket
that opens automatically in your browser.

Usage:
    python3 visual_simulation.py
"""

import json
import math
import os
import sys
import webbrowser
from collections import defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, '/Users/archie-raythompson/Sweepstake')
from wc2026_predictor import (
    GROUPS, R32_FIXED, R16_PAIRS, QF_PAIRS, SF_PAIRS,
    ELO_START, load_cache, load_data, load_known_results,
    expected_goals, sim_match, sim_knockout_match,
    pick_best_third, assign_third_place_teams,
    resolve_r32_bracket, build_next_round,
)

# combo indices → matchday (1-based) for 4-team group
# combinations order: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
ROUND_MAP = [1, 2, 3, 3, 2, 1]

OUT_PATH = '/tmp/wc2026_sim.html'


def build_strengths(regressions, current_elo):
    def strengths(team_a, team_b):
        elo_a = current_elo.get(team_a, ELO_START)
        elo_b = current_elo.get(team_b, ELO_START)
        return expected_goals(team_a, team_b, elo_a, elo_b, regressions)
    return strengths


def sim_one_tournament(strengths, current_elo, rng, known_results=None):
    """
    Run one full tournament simulation and return a detailed log of every match.
    Returns dict with 'groups' and 'knockout' sections for the HTML renderer.
    """
    known_group = (known_results or {}).get('group', {})
    known_ko    = (known_results or {}).get('knockout', {})

    # ── Group stage ───────────────────────────────────────────────────────────
    standings = {}
    for letter, members in GROUPS.items():
        for team in members:
            standings[team] = {'pts': 0, 'gf': 0, 'ga': 0, 'gd': 0, 'group': letter}

    group_data   = {}   # letter → {teams, rounds, standings}
    group_tables = {}

    for letter, members in GROUPS.items():
        # 3 matchday buckets, each a list of match dicts
        rounds = {1: [], 2: [], 3: []}

        for combo_idx, (h, a) in enumerate(combinations(members, 2)):
            matchday = ROUND_MAP[combo_idx]
            key = '|'.join(sorted([h, a]))

            if key in known_group:
                rec = known_group[key]
                if rec['home'] == h:
                    gh, ga = rec['home_score'], rec['away_score']
                else:
                    gh, ga = rec['away_score'], rec['home_score']
                is_known = True
            else:
                lam_h, lam_a = strengths(h, a)
                gh, ga = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
                is_known = False

            rounds[matchday].append({
                'home': h, 'away': a,
                'hg': gh, 'ag': ga,
                'known': is_known,
            })

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
                rng.random(),
                current_elo.get(t, ELO_START),
            ),
            reverse=True,
        )
        for rank, team in enumerate(ranked):
            standings[team]['group_rank'] = rank + 1
        group_tables[letter] = ranked

        group_data[letter] = {
            'teams': members,
            'rounds': [rounds[1], rounds[2], rounds[3]],
            'standings': [
                {
                    'team': t,
                    'pts':  standings[t]['pts'],
                    'gf':   standings[t]['gf'],
                    'ga':   standings[t]['ga'],
                    'gd':   standings[t]['gd'],
                    'rank': standings[t]['group_rank'],
                }
                for t in ranked
            ],
        }

    # ── Third-place assignment ────────────────────────────────────────────────
    ranked_thirds    = pick_best_third(standings, group_tables, current_elo, rng)
    third_assignment = assign_third_place_teams(ranked_thirds, standings)
    r32_matchups     = resolve_r32_bracket(group_tables, third_assignment)

    # ── Knockout rounds ───────────────────────────────────────────────────────
    def sim_ko_round_logged(matchups):
        results = []
        winners = {}
        for slot_id, t1, t2 in matchups:
            if slot_id in known_ko:
                winner = known_ko[slot_id]['winner']
                loser  = t2 if winner == t1 else t1
                results.append({
                    'slot': slot_id, 'home': t1, 'away': t2,
                    'hg': '?', 'ag': '?', 'winner': winner,
                    'pens': False, 'known': True,
                })
            else:
                lam1, lam2 = strengths(t1, t2)
                gh, ga = int(rng.poisson(lam1)), int(rng.poisson(lam2))
                if gh != ga:
                    winner = t1 if gh > ga else t2
                    pens   = False
                else:
                    # Draw — simulate penalty winner
                    p1 = lam1 / (lam1 + lam2 + 1e-10)
                    winner = t1 if rng.random() < p1 else t2
                    pens   = True
                results.append({
                    'slot': slot_id, 'home': t1, 'away': t2,
                    'hg': gh, 'ag': ga, 'winner': winner,
                    'pens': pens, 'known': False,
                })
            winners[slot_id] = winner
        return results, winners

    r32_results, r32_winners = sim_ko_round_logged(r32_matchups)
    r16_matchups = build_next_round(R16_PAIRS, r32_winners, 'R16-')
    r16_results, r16_winners = sim_ko_round_logged(r16_matchups)
    qf_matchups  = build_next_round(QF_PAIRS, r16_winners, 'QF')
    qf_results,  qf_winners  = sim_ko_round_logged(qf_matchups)
    sf_matchups  = build_next_round(SF_PAIRS, qf_winners, 'SF')
    sf_results,  sf_winners  = sim_ko_round_logged(sf_matchups)

    final_team1 = list(sf_winners.values())[0]
    final_team2 = list(sf_winners.values())[1]
    final_matchup = [('Final', final_team1, final_team2)]
    final_results, final_winners = sim_ko_round_logged(final_matchup)
    champion = final_winners['Final']

    return {
        'groups': group_data,
        'knockout': {
            'R32':   r32_results,
            'R16':   r16_results,
            'QF':    qf_results,
            'SF':    sf_results,
            'Final': final_results,
        },
        'champion': champion,
    }


def generate_html(sim_data):
    data_json = json.dumps(sim_data, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WC 2026 — Visual Simulation</title>
<style>
:root {{
  --bg:#0f1117; --surface:#1a1d27; --surface2:#22263a; --border:#2e3250;
  --accent:#4ade80; --accent2:#60a5fa; --gold:#fbbf24; --text:#e2e8f0;
  --muted:#64748b; --red:#f87171;
}}
*{{ box-sizing:border-box; margin:0; padding:0; }}
body{{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:var(--bg); color:var(--text); min-height:100vh; }}

/* ── Controls bar ── */
#controls{{
  position:fixed; top:0; left:0; right:0; z-index:100;
  background:var(--surface); border-bottom:1px solid var(--border);
  display:flex; align-items:center; gap:1rem; padding:0 1.5rem; height:52px;
}}
.logo{{ font-weight:700; color:var(--accent); font-size:1rem; margin-right:auto; }}
#phase-label{{ font-size:0.85rem; color:var(--muted); min-width:220px; text-align:center; }}
.btn{{
  background:var(--surface2); border:1px solid var(--border); color:var(--text);
  padding:0.35rem 0.9rem; border-radius:6px; cursor:pointer; font-size:0.85rem;
  transition:background 0.15s;
}}
.btn:hover{{ background:var(--border); }}
.btn.active{{ background:var(--accent); color:#000; border-color:var(--accent); }}
.speed-group{{ display:flex; gap:0.25rem; }}

/* ── Main layout ── */
#stage{{ padding: 72px 1rem 2rem; max-width:1400px; margin:0 auto; }}

/* ── Group grid ── */
#group-stage{{ display:none; }}
.groups-grid{{
  display:grid;
  grid-template-columns: repeat(4, 1fr);
  gap:0.75rem;
}}
@media(max-width:900px){{ .groups-grid{{ grid-template-columns:repeat(2,1fr); }} }}

.group-card{{
  background:var(--surface); border:1px solid var(--border); border-radius:10px;
  overflow:hidden;
}}
.group-header{{
  background:var(--surface2); padding:0.4rem 0.75rem;
  font-size:0.7rem; font-weight:700; text-transform:uppercase;
  letter-spacing:0.1em; color:var(--muted); border-bottom:1px solid var(--border);
}}
.group-table{{ width:100%; border-collapse:collapse; font-size:0.78rem; }}
.group-table th{{
  padding:0.25rem 0.5rem; text-align:right; color:var(--muted);
  font-size:0.65rem; text-transform:uppercase;
}}
.group-table th:first-child{{ text-align:left; }}
.group-table td{{ padding:0.3rem 0.5rem; text-align:right; border-top:1px solid var(--border); }}
.group-table td:first-child{{ text-align:left; font-weight:500; }}
.group-table tr.qualified td{{ color:var(--accent); }}
.group-table tr.third td{{ color:var(--accent2); }}
.group-table tr.eliminated td{{ color:var(--muted); }}

.match-results{{ padding:0.4rem 0.6rem; border-top:1px solid var(--border); }}
.match-line{{
  display:flex; justify-content:space-between; align-items:center;
  font-size:0.72rem; padding:0.15rem 0; color:var(--muted);
  opacity:0; transition:opacity 0.4s;
}}
.match-line.shown{{ opacity:1; }}
.match-line .score{{ color:var(--text); font-weight:700; margin:0 0.4rem; }}
.match-line .known-tag{{
  font-size:0.6rem; color:var(--accent); margin-left:4px;
}}

.matchday-header{{
  font-size:0.65rem; color:var(--muted); text-transform:uppercase;
  letter-spacing:0.08em; padding:0.3rem 0 0.1rem; border-top:1px solid var(--border);
}}
.matchday-header:first-child{{ border-top:none; padding-top:0; }}

/* ── Knockout bracket ── */
#knockout-stage{{ display:none; }}
.ko-title{{
  text-align:center; font-size:1.1rem; font-weight:700;
  color:var(--text); margin-bottom:1.5rem;
}}
.ko-title span{{ color:var(--muted); font-size:0.85rem; font-weight:400; margin-left:0.5rem; }}

.bracket{{
  display:grid;
  grid-template-columns: repeat(5, 1fr);
  gap:0.5rem;
  align-items:center;
}}
.round-col{{ display:flex; flex-direction:column; gap:0.4rem; }}
.round-label{{
  text-align:center; font-size:0.65rem; font-weight:700;
  text-transform:uppercase; letter-spacing:0.1em;
  color:var(--muted); margin-bottom:0.5rem;
}}

.match-card{{
  background:var(--surface); border:1px solid var(--border);
  border-radius:8px; overflow:hidden; font-size:0.75rem;
  transition:border-color 0.3s;
}}
.match-card.active{{ border-color:var(--accent); }}
.match-card.done{{ border-color:var(--border); }}

.match-team{{
  display:flex; justify-content:space-between; align-items:center;
  padding:0.3rem 0.5rem; border-bottom:1px solid var(--border);
}}
.match-team:last-child{{ border-bottom:none; }}
.match-team.winner{{ background:var(--surface2); color:var(--accent); font-weight:700; }}
.match-team.loser{{ color:var(--muted); }}
.team-goals{{ font-weight:700; font-size:0.8rem; }}
.pens-tag{{ font-size:0.6rem; color:var(--gold); margin-left:3px; }}
.match-slot{{ font-size:0.6rem; color:var(--muted); padding:0.15rem 0.5rem;
              background:var(--surface2); border-bottom:1px solid var(--border); }}

/* Final / champion */
.final-col{{
  display:flex; flex-direction:column; align-items:center; gap:1rem;
}}
.champion-card{{
  background:linear-gradient(135deg, var(--surface2), var(--surface));
  border:2px solid var(--gold); border-radius:12px; padding:1.2rem 1.5rem;
  text-align:center; min-width:140px;
  opacity:0; transition:opacity 0.5s;
}}
.champion-card.shown{{ opacity:1; }}
.champion-label{{ font-size:0.7rem; color:var(--gold); text-transform:uppercase;
                  letter-spacing:0.1em; margin-bottom:0.4rem; }}
.champion-name{{ font-size:1.1rem; font-weight:700; color:var(--text); }}
.trophy{{ font-size:2rem; margin-bottom:0.3rem; }}

/* ── Fade-in animation ── */
@keyframes fadeIn{{ from{{opacity:0;transform:translateY(6px)}} to{{opacity:1;transform:none}} }}
.fade-in{{ animation:fadeIn 0.4s ease forwards; }}
</style>
</head>
<body>

<div id="controls">
  <div class="logo">⚽ WC 2026 Simulation</div>
  <div id="phase-label">Ready</div>
  <div class="speed-group">
    <button class="btn" onclick="setSpeed('slow')">Slow</button>
    <button class="btn active" id="spd-normal" onclick="setSpeed('normal')">Normal</button>
    <button class="btn" onclick="setSpeed('fast')">Fast</button>
  </div>
  <button class="btn" id="play-btn" onclick="togglePlay()">▶ Play</button>
</div>

<div id="stage">
  <div id="group-stage"></div>
  <div id="knockout-stage"></div>
</div>

<script>
const SIM = {data_json};

const SPEEDS = {{ slow: 2800, normal: 1400, fast: 550 }};
let speed    = 'normal';
let playing  = false;
let timer    = null;
let stepFn   = null;

const GROUP_LETTERS = Object.keys(SIM.groups);
const KO_ROUNDS     = ['R32','R16','QF','SF','Final'];
const KO_LABELS     = {{ R32:'Round of 32', R16:'Round of 16', QF:'Quarter-Finals', SF:'Semi-Finals', Final:'Final' }};

// ── Utilities ────────────────────────────────────────────────────────────────

function setSpeed(s) {{
  speed = s;
  document.querySelectorAll('.speed-group .btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  if (playing) {{ clearTimeout(timer); scheduleNext(); }}
}}

function togglePlay() {{
  playing = !playing;
  document.getElementById('play-btn').textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) scheduleNext();
  else clearTimeout(timer);
}}

function scheduleNext() {{
  if (!stepFn) return;
  timer = setTimeout(() => {{
    const done = stepFn();
    if (done) {{ playing = false; document.getElementById('play-btn').textContent = '▶ Play'; }}
    else if (playing) scheduleNext();
  }}, SPEEDS[speed]);
}}

function setPhase(text) {{
  document.getElementById('phase-label').textContent = text;
}}

// ── Group stage ──────────────────────────────────────────────────────────────

function buildGroupStage() {{
  const container = document.getElementById('group-stage');
  container.style.display = 'block';
  container.innerHTML = '<div class="groups-grid" id="groups-grid"></div>';
  const grid = document.getElementById('groups-grid');

  GROUP_LETTERS.forEach(letter => {{
    const g = SIM.groups[letter];
    const card = document.createElement('div');
    card.className = 'group-card';
    card.id = `group-${{letter}}`;
    card.innerHTML = `
      <div class="group-header">Group ${{letter}}</div>
      <table class="group-table">
        <thead><tr>
          <th>Team</th><th>P</th><th>GD</th><th>Pts</th>
        </tr></thead>
        <tbody id="standings-${{letter}}">
          ${{g.teams.map(t => `<tr id="row-${{t.replace(/ /g,'_')}}">
            <td>${{t}}</td><td>0</td><td>0</td><td>0</td>
          </tr>`).join('')}}
        </tbody>
      </table>
      <div class="match-results" id="matches-${{letter}}"></div>
    `;
    grid.appendChild(card);
  }});
}}

// Running totals per team for the group stage
const runningStandings = {{}};

function initRunningStandings() {{
  GROUP_LETTERS.forEach(letter => {{
    SIM.groups[letter].teams.forEach(t => {{
      runningStandings[t] = {{ pts:0, gf:0, ga:0, gd:0, played:0 }};
    }});
  }});
}}

function updateStandingsDisplay(letter) {{
  const g     = SIM.groups[letter];
  const tbody = document.getElementById(`standings-${{letter}}`);
  // Sort by pts, gd, gf
  const teams = [...g.teams].sort((a,b) => {{
    const sa = runningStandings[a], sb = runningStandings[b];
    if (sb.pts !== sa.pts) return sb.pts - sa.pts;
    if (sb.gd  !== sa.gd)  return sb.gd  - sa.gd;
    return sb.gf - sa.gf;
  }});
  tbody.innerHTML = teams.map(t => {{
    const s = runningStandings[t];
    return `<tr id="row-${{t.replace(/ /g,'_')}}">
      <td>${{t}}</td>
      <td>${{s.played}}</td>
      <td>${{s.gd >= 0 ? '+' : ''}}${{s.gd}}</td>
      <td>${{s.pts}}</td>
    </tr>`;
  }}).join('');
}}

function showMatchday(matchdayIdx) {{
  // matchdayIdx: 0,1,2
  GROUP_LETTERS.forEach(letter => {{
    const g       = SIM.groups[letter];
    const matches = g.rounds[matchdayIdx];
    const box     = document.getElementById(`matches-${{letter}}`);

    const hdr = document.createElement('div');
    hdr.className = 'matchday-header';
    hdr.textContent = `Matchday ${{matchdayIdx + 1}}`;
    box.appendChild(hdr);

    matches.forEach(m => {{
      // Update running standings
      const h = runningStandings[m.home], a = runningStandings[m.away];
      h.gf += m.hg; h.ga += m.ag; h.gd = h.gf - h.ga; h.played++;
      a.gf += m.ag; a.ga += m.hg; a.gd = a.gf - a.ga; a.played++;
      if (m.hg > m.ag)      {{ h.pts += 3; }}
      else if (m.hg < m.ag) {{ a.pts += 3; }}
      else                   {{ h.pts += 1; a.pts += 1; }}

      // Render match line
      const line = document.createElement('div');
      line.className = 'match-line';
      const knownTag = m.known ? '<span class="known-tag">✓</span>' : '';
      line.innerHTML = `
        <span>${{m.home}}${{knownTag}}</span>
        <span class="score">${{m.hg}}–${{m.ag}}</span>
        <span>${{m.away}}${{knownTag}}</span>
      `;
      box.appendChild(line);
      setTimeout(() => line.classList.add('shown'), 50);

      updateStandingsDisplay(letter);
    }});
  }});
}}

function finaliseGroupColours() {{
  GROUP_LETTERS.forEach(letter => {{
    const g      = SIM.groups[letter];
    const ranked = g.standings;
    ranked.forEach((row, i) => {{
      const id = `row-${{row.team.replace(/ /g,'_')}}`;
      const tr = document.getElementById(id);
      if (!tr) return;
      if (i < 2)      tr.className = 'qualified';
      else if (i === 2) tr.className = 'third';
      else              tr.className = 'eliminated';
    }});
    // Replace running standings with final official standings
    const tbody = document.getElementById(`standings-${{letter}}`);
    tbody.innerHTML = ranked.map((row, i) => {{
      const cls = i < 2 ? 'qualified' : i === 2 ? 'third' : 'eliminated';
      return `<tr class="${{cls}}">
        <td>${{row.team}}</td>
        <td>3</td>
        <td>${{row.gd >= 0 ? '+' : ''}}${{row.gd}}</td>
        <td>${{row.pts}}</td>
      </tr>`;
    }}).join('');
  }});
}}

// ── Knockout bracket ─────────────────────────────────────────────────────────

function buildBracket() {{
  const ks = document.getElementById('knockout-stage');
  ks.style.display = 'block';
  ks.innerHTML = `
    <div class="ko-title">Knockout Stage <span>results reveal round by round</span></div>
    <div class="bracket" id="bracket-grid"></div>
  `;
  const grid = document.getElementById('bracket-grid');

  KO_ROUNDS.forEach(round => {{
    const col = document.createElement('div');
    col.className = 'round-col';
    col.id = `col-${{round}}`;

    const lbl = document.createElement('div');
    lbl.className = 'round-label';
    lbl.textContent = KO_LABELS[round];
    col.appendChild(lbl);

    if (round === 'Final') {{
      const fc = document.createElement('div');
      fc.className = 'final-col';
      fc.id = 'final-col';
      fc.innerHTML = `
        <div class="champion-card" id="champion-card">
          <div class="trophy">🏆</div>
          <div class="champion-label">World Champion</div>
          <div class="champion-name" id="champion-name">—</div>
        </div>
      `;
      col.appendChild(fc);
    }}

    SIM.knockout[round].forEach((m, i) => {{
      const card = document.createElement('div');
      card.className = 'match-card';
      card.id = `match-${{m.slot}}`;
      card.innerHTML = `
        <div class="match-slot">${{m.slot}}</div>
        <div class="match-team" id="home-${{m.slot}}"><span>TBD</span><span class="team-goals">—</span></div>
        <div class="match-team" id="away-${{m.slot}}"><span>TBD</span><span class="team-goals">—</span></div>
      `;
      col.appendChild(card);
    }});

    grid.appendChild(col);
  }});

  // Pre-fill team names (scores hidden until revealed)
  KO_ROUNDS.forEach(round => {{
    SIM.knockout[round].forEach(m => {{
      const hEl = document.getElementById(`home-${{m.slot}}`);
      const aEl = document.getElementById(`away-${{m.slot}}`);
      if (hEl) hEl.querySelector('span').textContent = m.home;
      if (aEl) aEl.querySelector('span').textContent = m.away;
    }});
  }});
}}

function revealKoRound(round) {{
  SIM.knockout[round].forEach(m => {{
    const card = document.getElementById(`match-${{m.slot}}`);
    const hEl  = document.getElementById(`home-${{m.slot}}`);
    const aEl  = document.getElementById(`away-${{m.slot}}`);
    if (!card || !hEl || !aEl) return;

    card.classList.add('active');
    setTimeout(() => {{
      const hGoals = m.known ? '✓' : m.hg;
      const aGoals = m.known ? '✓' : m.ag;
      const pTag   = m.pens ? '<span class="pens-tag">pen</span>' : '';

      hEl.querySelector('.team-goals').innerHTML = `${{hGoals}}${{m.winner === m.home ? pTag : ''}}`;
      aEl.querySelector('.team-goals').innerHTML = `${{aGoals}}${{m.winner === m.away ? pTag : ''}}`;

      if (m.winner === m.home) {{
        hEl.classList.add('winner'); aEl.classList.add('loser');
      }} else {{
        aEl.classList.add('winner'); hEl.classList.add('loser');
      }}
      card.classList.remove('active');
      card.classList.add('done');
    }}, SPEEDS[speed] * 0.6);
  }});

  if (round === 'Final') {{
    setTimeout(() => {{
      document.getElementById('champion-name').textContent = SIM.champion;
      document.getElementById('champion-card').classList.add('shown');
    }}, SPEEDS[speed] * 0.9);
  }}
}}

// ── Main sequence ────────────────────────────────────────────────────────────

function run() {{
  buildGroupStage();
  initRunningStandings();

  let phase = 'group';
  let matchday = 0;
  let koRoundIdx = 0;

  stepFn = () => {{
    if (phase === 'group') {{
      if (matchday < 3) {{
        setPhase(`Group Stage — Matchday ${{matchday + 1}} / 3`);
        showMatchday(matchday);
        matchday++;
        return false;
      }} else {{
        // Finalise group colours, transition to knockout
        finaliseGroupColours();
        setTimeout(() => {{
          document.getElementById('group-stage').style.display = 'none';
          buildBracket();
          phase = 'knockout';
          if (playing) scheduleNext();
        }}, SPEEDS[speed] * 1.5);
        return false;
      }}
    }}

    if (phase === 'knockout') {{
      if (koRoundIdx < KO_ROUNDS.length) {{
        const round = KO_ROUNDS[koRoundIdx];
        setPhase(KO_LABELS[round]);
        revealKoRound(round);
        koRoundIdx++;
        if (koRoundIdx >= KO_ROUNDS.length) {{
          setPhase(`🏆 ${{SIM.champion}} are World Champions!`);
          return true;
        }}
        return false;
      }}
      return true;
    }}
    return true;
  }};

  setPhase('Press Play to start');
}}

run();
</script>
</body>
</html>"""


if __name__ == '__main__':
    print("Loading model data...")
    _, current_elo = load_data()

    regressions = load_cache()
    if regressions is None:
        print("No cache found — run python3 wc2026_predictor.py first to build the cache.")
        sys.exit(1)

    print("Loaded regressions from cache.")
    strengths     = build_strengths(regressions, current_elo)
    known_results = load_known_results()
    rng           = np.random.default_rng()  # random seed each run for variety

    n_group = len(known_results.get('group', {}))
    n_ko    = len(known_results.get('knockout', {}))
    if n_group or n_ko:
        print(f"Using {n_group} known group result(s) and {n_ko} knockout result(s).")

    print("Running simulation...")
    sim_data = sim_one_tournament(strengths, current_elo, rng, known_results)
    print(f"Champion: {sim_data['champion']}")

    html = generate_html(sim_data)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Opening simulation in browser... ({OUT_PATH})")
    webbrowser.open(f'file://{OUT_PATH}')
