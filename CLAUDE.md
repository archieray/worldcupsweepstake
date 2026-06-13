# WC 2026 Sweepstake Predictor — CLAUDE.md

## Project Overview

A Monte Carlo tournament simulator for the 2026 FIFA World Cup, built around a sweepstake with 12 players (4 teams each). The model predicts stage probabilities for all 48 teams and scores each player's expected points.

## Key Files

| File | Purpose |
|---|---|
| `wc2026_predictor.py` | Main model — ELO computation, Poisson regression, 100k-sim Monte Carlo |
| `update.py` | CLI to record a match result and trigger re-simulation |
| `standings.py` | Print sweepstake standings from predictions CSV |
| `sweepstake.json` | Player → team assignments |
| `match_results.json` | Known match results (written by `update.py`) |
| `wc2026_predictions.csv` | Model output — stage probabilities per team |
| `wc2026_cache.json` | Disk cache of fitted regressions (invalidated by `results.csv` mtime) |
| `results.csv` | 49,477 international matches — model training data |
| `teams.csv` | 48 WC 2026 teams with current ELO |

## How to Run

```bash
cd /Users/archie-raythompson/Sweepstake

# Run full simulation (uses cache if results.csv unchanged)
python3 wc2026_predictor.py

# Record a group result and re-simulate
python3 update.py group "Spain" 3 "Cape Verde" 0

# Record a knockout result
python3 update.py knockout R32-1 "South Korea" 0 "Canada" 2

# Knockout that went to penalties (level score)
python3 update.py knockout R32-1 "Spain" 1 "France" 1 --winner "Spain"

# Remove a result
python3 update.py remove group "Spain" "Cape Verde"
python3 update.py remove knockout R32-1

# Show sweepstake standings
python3 standings.py
```

## Model Architecture

**Per-team weighted Poisson regression** (Gilch 2022):
- Two models per team: goals scored and goals conceded
- Covariates: opponent ELO at match time + location (+1 home, 0 neutral, -1 away)
- `log μ = a0 + a1·elo_opp + a2·location`
- Final λ = average of scored-model and conceded-model predictions
- Training data: all international matches since 2016, recency-weighted (3-year half-life)
- All WC simulated matches use `location = 0` (neutral venues)

**Simulation:**
- 100,000 Monte Carlo runs, fixed bracket structure
- Group stage goals vectorised: `rng.poisson(np.outer(lams, np.ones(n_sims)))` — (72, N_SIMS) drawn at once
- Known results locked in via `match_results.json` — not re-simulated

**Cache:** `wc2026_cache.json` stores fitted regression coefficients keyed by `results.csv` mtime. Invalidated automatically when results change. Delete it manually to force a refit.

## Sweepstake Scoring

| Stage | Points |
|---|---|
| Group exit | 0 |
| Round of 32 | 1 |
| Round of 16 | 3 |
| Quarter-final | 6 |
| Semi-final | 10 |
| 3rd place / Final | 15 |
| Winner | 25 |

## Players & Teams

| Player | Teams |
|---|---|
| Archie | Spain, Mexico, Egypt, Ghana |
| Ben | Argentina, Senegal, Algeria, Uzbekistan |
| Dan | Netherlands, Australia, DR Congo, Qatar |
| Hywel | Croatia, Austria, Czechia, Iraq |
| Jon | England, Switzerland, Canada, Cape Verde |
| Krish | France, Iran, Panama, Saudi Arabia |
| Lucca | Germany, Türkiye, Norway, South Africa |
| Toby | Morocco, Japan, Ivory Coast, New Zealand |
| Will H | Portugal, United States, Sweden, Curaçao |
| Will P | Colombia, Uruguay, Paraguay, Haiti |
| Will S | Brazil, South Korea, Scotland, Jordan |
| Will T | Belgium, Ecuador, Tunisia, Bosnia and Herzegovina |

## Canonical Team Names

These differ from FIFA/results.csv names — always use these:

| Use this | Not this |
|---|---|
| South Korea | Korea Republic |
| Ivory Coast | Côte d'Ivoire |
| Iran | IR Iran |
| Bosnia and Herzegovina | Bosnia-Herzegovina |
| DR Congo | Congo DR |
| Türkiye | Turkey |

## JSON Structures

**`match_results.json` group key:** `'|'.join(sorted([team_a, team_b]))` — order-independent.

**`match_results.json` knockout key:** slot ID (e.g. `R32-1`, `R16-3`, `QF2`, `SF1`, `Final`).

Valid knockout slots: `R32-1..R32-16`, `R16-1..R16-8`, `QF1..QF4`, `SF1`, `SF2`, `Final`.

## Groups

```
A: Mexico, South Korea, South Africa, Czechia
B: Canada, Bosnia and Herzegovina, Qatar, Switzerland
C: Brazil, Morocco, Haiti, Scotland
D: United States, Paraguay, Australia, Türkiye
E: Germany, Curaçao, Ivory Coast, Ecuador
F: Netherlands, Japan, Sweden, Tunisia
G: Belgium, Egypt, Iran, New Zealand
H: Spain, Cape Verde, Saudi Arabia, Uruguay
I: France, Senegal, Iraq, Norway
J: Argentina, Algeria, Austria, Jordan
K: Portugal, DR Congo, Uzbekistan, Colombia
L: England, Croatia, Ghana, Panama
```
