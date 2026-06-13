# WC 2026 Sweepstake Predictor

A Monte Carlo tournament simulator for the 2026 FIFA World Cup, built around a 12-player sweepstake (4 teams each). Uses per-team Poisson regression fitted on international match history to predict stage probabilities and score each player's expected points.

## Features

- **100,000-simulation Monte Carlo model** — per-team Poisson regression with ELO and location covariates (Gilch 2022)
- **Live result locking** — record real match results via CLI; the simulator fixes those outcomes and re-runs
- **Web dashboard** — Flask app with sweepstake standings, team stats (live via API), and upcoming fixtures with win probabilities
- **Visual simulation** — animated bracket that plays through one full tournament simulation in the browser

## Setup

```bash
git clone https://github.com/archieray/worldcupsweepstake.git
cd worldcupsweepstake
pip install flask requests python-dotenv numpy scipy
```

Get a free API key from [football-data.org](https://www.football-data.org/client/register) for live fixtures and top scorers, then set it in your environment:

```bash
export FOOTBALL_DATA_API_KEY=your_key_here
```

## Usage

### Run the web dashboard
```bash
python3 app.py
# Open http://localhost:5001
```

### Record a match result and re-simulate
```bash
# Group stage
python3 update.py group "Spain" 3 "Cape Verde" 0

# Knockout (winner from score)
python3 update.py knockout R32-1 "South Korea" 0 "Canada" 2

# Knockout that went to penalties
python3 update.py knockout R32-1 "Spain" 1 "France" 1 --winner "Spain"

# Remove a result
python3 update.py remove group "Spain" "Cape Verde"
python3 update.py remove knockout R32-1
```

### Run the visual simulation
```bash
python3 visual_simulation.py
# Opens an animated bracket in your browser — different outcome every run
```

### Print standings in the terminal
```bash
python3 standings.py
```

### Run the model directly
```bash
python3 wc2026_predictor.py
```

## Model

Per-team weighted Poisson regression: two models per team (goals scored, goals conceded) with opponent ELO and match location as covariates. Fitted on all international matches since 2016 with a 3-year recency half-life. Regression coefficients are cached to `wc2026_cache.json` and reused until `results.csv` changes.

Simulation uses the correct FIFA 2026 fixed bracket structure including best-8 third-place seeding for the Round of 32.

## Sweepstake Scoring

| Stage | Points |
|---|---|
| Group exit | 0 |
| Round of 32 | 1 |
| Round of 16 | 3 |
| Quarter-final | 6 |
| Semi-final | 10 |
| Final | 15 |
| Winner | 25 |

## Players

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
