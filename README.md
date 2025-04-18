# 🏀 NBA Season Simulation Using Parallel Markov Chains

This project simulates full NBA seasons using team-specific transition probabilities derived from 27 years of play-by-play data (1997–2023). The system models game flow as a Markov process, predicting season win totals using parallelized data processing, transition matrix generation, and team-level simulations. Built for CMSE 401 (Parallel Computing) at Michigan State University.

## Project Overview

The pipeline consists of four main stages, each designed for efficient parallel computation:

1. `year_processing.py`  
   - Loads and cleans raw play-by-play CSVs by season  
   - Removes irrelevant events (e.g., substitutions), normalizes data structure  
   - Runs across seasons using `ProcessPoolExecutor` with 4 threads per process

2. `pbp_processing.py`  
   - Adds possession tracking, inbound insertion, shot type correction, and event labeling  
   - Identifies possession boundaries and updates rows with contextual features  
   - Also parallelized per season with 4-thread processes

3. `team_probabilities.py`  
   - Builds team-specific Markov transition matrices and possession start distributions  
   - Computes empirical frequencies of possession-type transitions split by home and away  
   - Process-level parallelism (no threading) across 30 teams and recent seasons

## Running the Simulation (Local or HPCC)

### Option 1: Local Execution

```bash
python year_processing.py
python pbp_processing.py
python team_probabilities.py
```

Then launch Jupyter and run:

```python
from simulation import run_simulation
run_simulation(learn_rate=0.53)
```

### Option 2: MSU HPCC Job Submission

Submit with:

```bash
sbatch data_cleaning.sb
```

Then submit with:

```bash
sbatch team_probabilities.sb
```

## Key Results

- Over 50% of NBA teams had a prediction error within ±4 wins  
- Best learning rate: 0.53  
- RMSE consistently declined over iterations, showing strong convergence

## Performance and Speedup

| Stage                   | Local Time (4 Proc × 4 Threads) | HPCC Time (28 Proc × 4 Threads) | Speedup |
|-------------------------|----------------------------------|----------------------------------|---------|
| `year_processing.py`    | 83 seconds                       | ~15 seconds                      | ~5.5×   |
| `pbp_processing.py`     | 538 seconds                      | ~77 seconds                      | ~7.0×   |
| `team_probabilities.py` | 137 seconds                      | ~41 seconds                      | ~3.3×   |

## Setup Instructions

```bash
git clone https://github.com/yourusername/nba-season-simulation.git
cd nba-season-simulation
```

Install dependencies:

```bash
pip install pandas numpy matplotlib tqdm
```

Then run each script or submit with `sbatch`, and launch the simulation:

```python
from simulation import run_simulation
run_simulation(learn_rate=0.53)
```

## References

- NBA Play-by-Play Dataset via Kaggle: https://www.kaggle.com  
- Buckets Shot Chart by Peter Beshai: https://peterbeshai.com/buckets/  
- OpenAI ChatGPT (GPT-4) — code guidance, formatting, and analysis  
- CMSE 401 (Parallel Computing) – Michigan State University

## Author

Trenton Smiley  
B.S. in Data Science, Class of 2026  
Michigan State University  
📧 trenton.smiley@msu.edu

