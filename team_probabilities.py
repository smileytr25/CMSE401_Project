import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time 
import json
from collections import defaultdict

turnover_events = [
    "5 Second Violation", "8 Second Violation", "Bad Pass Turnover", "Double Dribble",
    "Double Lane Violation", "Lost Ball Turnover", "Offensive 3 Second Violation", "Over And Back",
    "Traveling Violation", "Shot Clock Violation", "Offensive Goaltending", "Offensive Lane Violation",
    "Illegal Screen", "Inbound Turnover", "Charging Foul", "Palming Turnover",
    "Basket From Below Violation", "Illegal Assist Turnover", "Out Of Bounds"
]

foul_events = [
    "Away From The Play Foul", "Clear Path Foul", "Defensive 3 Second Violation", "Defensive Lane Violation",
    "Double Personal Foul", "Double Technical Foul", "Flagrant 1 Foul", "Inbound Foul",
    "Loose Ball Foul", "Personal Foul", "Shooting Foul", "Technical Foul",
    "Flagrant 2 Foul", "Transition Take Foul", "General Ejection", "Defensive Kicked Ball Violation",
    "Delay Of Game Technical", "Unsportsmanlike Technical", "Delay Of Game Violation"
]

free_throw_events = [
    "Clear Path Free Throw", "Flagrant Foul Free Throw 1 of 2",
    "Free Throw 1 of 1", "Free Throw 1 of 2", "Free Throw 1 of 3", "Free Throw 2 of 2",
    "Free Throw 2 of 3", "Free Throw 3 of 3", "Technical Foul Free Throw", "Flagrant Foul Free Throw 1 of 1",
    "Flagrant Foul Free Throw 2 of 2", "Flagrant Foul Free Throw 2 of 3", "Flagrant Foul Free Throw 3 of 3"
]

def create_team_markov(group):
    name, df = group
    df = df.copy()

    df["location"] = np.where(df["homeTeam"] == name, "Home", "Away")
    df = df.dropna(subset=["possessionId"])

    possession_grouped = df.groupby(["gameid", "possessionId", "location"])

    transitions = []
    start_counts = {"Home": defaultdict(int), "Away": defaultdict(int)}
    possession_counts = {"Home": 0, "Away": 0}
    games = {"Home": set(), "Away": set()}

    for (gameid, poss_id, location), poss_df in possession_grouped:
        start_event = poss_df.iloc[0]["eventType"]
        end_event = poss_df.iloc[-1]["eventType"]

        games[location].add(gameid)  # Track unique games

        possession_counts[location] += 1
        start_counts[location][start_event] += 1

        if end_event in ["Inbound", "Defensive Rebound", "Steal", "Offensive Rebound", "Foul"]:
            end_event = "Turnover"

        transitions.append((location, start_event, end_event))

    transitions_df = pd.DataFrame(transitions, columns=["location", "start", "end"])

    transition_counts = (
        transitions_df.groupby(["location", "start", "end"])
        .size()
        .reset_index(name="count")
    )

    markov = {"Home": {"Counts": {}, "Totals": {}}, "Away": {"Counts": {}, "Totals": {}}}

    for loc in ["Home", "Away"]:
        df_loc = transition_counts[transition_counts["location"] == loc]
        pivot = df_loc.pivot(index="start", columns="end", values="count").fillna(0)
        totals = (pivot.T / pivot.sum(axis=1)).T.fillna(0)
        markov[loc]["Counts"] = pivot
        markov[loc]["Totals"] = totals.loc[["Defensive Rebound", "Inbound", "Steal"], ["2PT Blocked", "2PT Made", "2PT Missed", "3PT Made", "3PT Missed", "Free Throw", "Turnover"]]

    return name, markov, possession_counts, start_counts, {loc: len(games[loc]) for loc in ["Home", "Away"]}

def generate_season_markovs(season):
    df = pd.read_csv(f"data/pbp_{season}.csv")
    df = df[((~df["eventType"].str.contains("Quarter")) & 
             (~df["eventType"].str.contains("Timeout")) & 
             (~df["eventType"].str.contains("Coach Challenge")) &
             (~df["eventType"].str.contains("Instant Replay")))]

    df.loc[df["eventType"].isin(turnover_events), "eventType"] = "Turnover"
    df.loc[df["eventType"].isin(free_throw_events), "eventType"] = "Free Throw"
    df.loc[df["eventType"].isin(foul_events), "eventType"] = "Foul"

    groups = df.groupby("possessionTeam")
    team_markovs = {}
    team_meta = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(create_team_markov, group) for group in groups]
        for f in as_completed(futures):
            team, markov, poss_counts, start_counts, game_counts = f.result()
            team_markovs[team] = markov

            # Aggregate possession and start counts for metadata
            team_meta[team] = {
                "Home": {
                    "possessions": poss_counts["Home"],
                    "starts": dict(start_counts["Home"]),
                    "games": game_counts["Home"]
                },
                "Away": {
                    "possessions": poss_counts["Away"],
                    "starts": dict(start_counts["Away"]),
                    "games": game_counts["Away"]
                }
            }


    return season, team_markovs, team_meta

def parallel_main():
    seasons = list(range(2015, 2023))
    markovs = {}
    meta = {}

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(generate_season_markovs, season) for season in seasons]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Seasons"):
            season, team_markovs, team_meta = f.result()
            markovs[season] = team_markovs

            for team in team_meta:
                if team not in meta:
                    meta[team] = {"Home": {"possessions": 0, "starts": defaultdict(int), "games": 0},
                                  "Away": {"possessions": 0, "starts": defaultdict(int), "games": 0}}
                meta[team]["Home"]["possessions"] += team_meta[team]["Home"]["possessions"]
                meta[team]["Away"]["possessions"] += team_meta[team]["Away"]["possessions"]
                meta[team]["Home"]["games"] += team_meta[team]["Home"]["games"]
                meta[team]["Away"]["games"] += team_meta[team]["Away"]["games"]

                for k, v in team_meta[team]["Home"]["starts"].items():
                    meta[team]["Home"]["starts"][k] += v
                for k, v in team_meta[team]["Away"]["starts"].items():
                    meta[team]["Away"]["starts"][k] += v

    return markovs, meta

def average_markov_matrices(markovs):
    averaged_markovs = {}

    for team in set(t for season in markovs.values() for t in season):
        home_matrices = []
        away_matrices = []

        for season in markovs:
            if team in markovs[season]:
                markov = markovs[season][team]
                home_matrices.append(markov["Home"]["Totals"])
                away_matrices.append(markov["Away"]["Totals"])

        if home_matrices:
            home_avg = pd.concat(home_matrices).groupby(level=0).mean()
            away_avg = pd.concat(away_matrices).groupby(level=0).mean()

            averaged_markovs[team] = {
                "Home": home_avg.to_dict(),
                "Away": away_avg.to_dict(),
            }

    return averaged_markovs

def convert_meta_to_probs(meta):
    meta_out = {}

    for team, data in meta.items():
        meta_out[team] = {}

        for loc in ["Home", "Away"]:
            start_counts = data[loc]["starts"]
            total = sum(start_counts.values())
            if total > 0:
                probs = {k: v / total for k, v in start_counts.items()}
            else:
                probs = {}

            meta_out[team][loc] = {
                "avg_possessions_per_game": data[loc]["possessions"] / data[loc]["games"] if data[loc]["games"] else 0,
                "start_type_probs": probs
            }

    return meta_out

if __name__ == "__main__":
    start_time = time.time()
    markovs, meta = parallel_main()

    team_matrices = average_markov_matrices(markovs)
    team_metadata = convert_meta_to_probs(meta)

    with open("team_matrices.json", "w") as f:
        json.dump(team_matrices, f, indent=2)

    with open("team_metadata.json", "w") as f:
        json.dump(team_metadata, f, indent=2)

    print("Done in", round(time.time() - start_time, 2), "seconds.")
