import json
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

actual_wins_2223 = {
    "ATL": 41, "BOS": 57, "BKN": 45, "CHA": 27, "CHI": 40,
    "CLE": 51, "DAL": 38, "DEN": 53, "DET": 17, "GSW": 44,
    "HOU": 22, "IND": 35, "LAC": 44, "LAL": 43, "MEM": 51,
    "MIA": 44, "MIL": 58, "MIN": 42, "NOP": 42, "NYK": 47,
    "OKC": 40, "ORL": 34, "PHI": 54, "PHX": 45, "POR": 33,
    "SAC": 48, "SAS": 22, "TOR": 41, "UTA": 37, "WAS": 35
}



def flip_nested_dict(d):
    flipped = defaultdict(dict)
    for outer, inner in d.items():
        for k, v in inner.items():
            flipped[k][outer] = v
    return dict(flipped)

def simulate_possession(location, team_matrix, team_meta):
    start_probs = team_meta[location]["start_type_probs"].copy()
    for key in list(start_probs.keys()):
        if "Jump Ball" in key:
            del start_probs[key]

    start_type = random.choices(
        list(start_probs.keys()), 
        list(start_probs.values())
    )[0]

    transition_probs = flip_nested_dict(team_matrix[location]).get(start_type, {}) 
    outcome = random.choices(
        list(transition_probs.keys()),
        list(transition_probs.values())
    )[0]

    if outcome in ["2PT Made", "3PT Made"]:
        return 2 if outcome == "2PT Made" else 3
    elif outcome == "Free Throw":
        return 1
    else:
        return 0

def monte_carlo_sim(location, probs, meta, poss):
    return np.mean([simulate_possession(location, probs, meta) for _ in range(int(poss))])

def simulate_game(team1, team2, home_team, team1_matrix, team2_matrix, team1_meta, team2_meta):
    if home_team == team1:
        team1_loc, team2_loc = "Home", "Away"
    else:
        team1_loc, team2_loc = "Away", "Home"

    poss = (0.75 * team1_meta[team1_loc]['avg_possessions_per_game'] +
            0.25 * team2_meta[team2_loc]['avg_possessions_per_game'])

    team1_score = monte_carlo_sim(team1_loc, team1_matrix, team1_meta, poss)
    team2_score = monte_carlo_sim(team2_loc, team2_matrix, team2_meta, poss)
    return team1 if team1_score > team2_score else team2

def backpropagate_possessions(
    schedule, actual_wins, meta, probs, simulate_game,
    iterations=15, learn_rate=1.0
):
    meta = copy.deepcopy(meta)
    rmse = 15
    best_rmse = float("inf")
    best_iteration = 0
    best_meta = None
    rmses = []

    for i in range(iterations):
        records = {team: [0, 0] for team in actual_wins}

        for idx, (gameid, homeTeam, awayTeam) in schedule.iterrows():
            if sum(records[homeTeam]) > 81 or sum(records[awayTeam]) > 81:
                continue

            winner = simulate_game(
                homeTeam, awayTeam, homeTeam,
                probs[homeTeam], probs[awayTeam],
                meta[homeTeam], meta[awayTeam]
            )

            if winner == homeTeam:
                records[homeTeam][0] += 1
                records[awayTeam][1] += 1
            else:
                records[awayTeam][0] += 1
                records[homeTeam][1] += 1

        for team in records:
            predicted_wins = records[team][0]
            error = predicted_wins - actual_wins[team]
            threshold = min(max(rmse * 1.5, 8), 20)
            if abs(error) > threshold:
                for loc in ["Home", "Away"]:
                    for start in ["Defensive Rebound", "Steal", "Inbound"]:
                        for outcome in ["2PT Made", "3PT Made", "Free Throw"]:
                            probs[team][loc][outcome][start] *= (1 - error * learn_rate * 0.01)
                        for outcome in ["Turnover", "2PT Missed", "3PT Missed"]:
                            probs[team][loc][outcome][start] *= (1 + error * learn_rate * 0.01)
    
                # Normalize the transition matrix after modifying it
                for loc in ["Home", "Away"]:
                    transposed = flip_nested_dict(probs[team][loc])
                    for start_type, outcomes in transposed.items():
                        total = sum(outcomes.values())
                        if total > 0:
                            for outcome in outcomes:
                                outcomes[outcome] /= total
                    probs[team][loc] = flip_nested_dict(transposed)
        
                # Also tune possession counts
                meta[team]["Home"]['avg_possessions_per_game'] -= error * learn_rate
                meta[team]["Away"]['avg_possessions_per_game'] -= error * learn_rate


        errors = [records[team][0] - actual_wins[team] for team in records]
        rmse = np.sqrt(np.mean(np.square(errors)))
        rmses.append(rmse)
        learn_rate = 1 / (i + 2) ** 1.25

        if rmse < best_rmse:
            best_rmse = rmse
            best_meta = copy.deepcopy(meta)
            best_iteration = i

    predicts = []
    actuals = []
    for team in sorted(records):
        predicted = records[team][0]
        actual = actual_wins[team]
        predicts.append(predicted)
        actuals.append(actual)

    return best_meta, best_rmse, best_iteration, predicts, actuals, rmses 

def run_simulation(iterations=15, learn_rate=1.0):
    df = pd.read_csv("data/pbp_2023.csv")
    schedule = (
        df[["gameid", "homeTeam", "awayTeam"]]
        .drop_duplicates(subset=["gameid"])
        .reset_index(drop=True)
    )
    
    with open("team_metadata.json", "r") as f:
        meta = json.load(f)
    
    with open("team_matrices.json", "r") as f:
        probs = json.load(f)
    
    best_meta, best_rmse, best_iteration, predicts, actuals, rmses = backpropagate_possessions(schedule, actual_wins_2223, meta, probs, simulate_game, iterations=15, learn_rate=1.0)
    return best_meta, best_rmse, best_iteration, predicts, actuals, rmses