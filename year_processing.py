#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:37:57 2025

@author: trentonsmiley
"""
import kagglehub
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import seaborn as sns
import time
from numba import njit

@njit
def iterate_indices_pts(indices, home_pts, away_pts):
    home_last_seen_data, away_last_seen_data = 0, 0
    for ind in indices:
        if np.isnan(home_pts[ind]) or home_pts[ind] < home_last_seen_data:
            home_pts[ind] = home_last_seen_data
        else:
            home_last_seen_data = home_pts[ind]
        
        if np.isnan(away_pts[ind]) or away_pts[ind] < away_last_seen_data:
            away_pts[ind] = away_last_seen_data
        else:
            away_last_seen_data = away_pts[ind]
        
    return home_pts, away_pts

def iterate_indices_def(ind, types, desc):
    if not isinstance(types[ind], str):
        if isinstance(desc[ind], str): 
             if "BLOCK" in desc[ind]:
                 types[ind] = "Block"
             elif "STEAL" in desc[ind]:
                 types[ind] = "Steal"
    return types

def iterate_indices_reb(ind, types, subtypes, players, desc, plyr_rebounds):
    if types[ind] == "Rebound":
        if players[ind] not in plyr_rebounds:
            plyr_rebounds[players[ind]] = [0, 0]
        split = desc[ind].split(":")
        off_reb = int(split[1].split(" ")[0])
        def_reb = int(split[2].split(')')[0])
        if plyr_rebounds[players[ind]][0] < off_reb:
            subtypes[ind] = "Offensive"
            plyr_rebounds[players[ind]][0] += 1
        else:
            subtypes[ind] = "Defensive"
            plyr_rebounds[players[ind]][1] += 1
    return subtypes, plyr_rebounds

def iterate_indices_jumpball(ind, types, results, teams):
    if types[ind] == "Jump Ball":
        if ind + 1 < len(teams):
            results[ind] = teams[ind + 1]
        else:
            results[ind] = "End of Game"
    return results 

def iterate_indices(indices, types, subtypes, results, players, teams, desc):
    plyr_rebounds = {}
    types, subtypes, results, players, teams, desc = list(types), list(subtypes), list(results), list(players), list(teams), list(desc)
    for ind in indices:
        types = iterate_indices_def(ind, types, desc)
        subtypes, plyr_rebounds = iterate_indices_reb(ind, types, subtypes, players, desc, plyr_rebounds)
        results = iterate_indices_jumpball(ind, types, results, teams)
    return types, subtypes, results

def process_group(gameId, group):
    group = group[(~group["team"].isna()) & (group["type"] != "Substitution")].copy()
    indices = np.arange(0, len(group), 1)
    h_pts, a_pts = iterate_indices_pts(indices, np.array(group["h_pts"]), np.array(group["a_pts"]))
    types, subtypes, results = iterate_indices(indices, group["type"], group['subtype'],  group["result"], group["player"], group["team"], group["desc"])

    group.loc[:, "h_pts"] = h_pts
    group.loc[:, "a_pts"] = a_pts
    group.loc[:, "type"] = types
    group.loc[:, "subtype"] = subtypes
    group.loc[:, "result"] = results

    group = group.fillna("None")
    return group

def parallel_process_season(path, year):
    df = pd.read_csv(path + f'/pbp{year}.csv')
    
    df["minutes_remaining"] = (df["clock"].apply(lambda x: x.split("PT")[1].split("M")[0])).astype("int64")
    df["seconds_remaining"] = (df["clock"].apply(lambda x: x.split("M")[1].split("S")[0])).astype("float64")
    df = df.drop("clock", axis=1)
    
    groups = df.groupby("gameid")
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_group, gid, grp) for gid, grp in groups]
        for f in as_completed(futures):
            results.append(f.result())
    return pd.concat(results, axis=0) 

def parallel_main():

    path = kagglehub.dataset_download("szymonjwiak/nba-play-by-play-data-1997-2023")
    years = list(range(1997, 2024))
    filenames = [f"./data/pbp_{year}.csv" for year in years]
    
    all_results = []
    with ProcessPoolExecutor(max_workers=27) as executor:
        futures = [executor.submit(parallel_process_season, path, year) for year in years]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Seasons"):
            all_results.append(f.result())
    return filenames, all_results

if __name__ == "__main__":
   start_time = time.time()
   filenames, results = parallel_main()
   for filename, result in zip(filenames, results):
       result.to_csv(filename)
   print(time.time() - start_time)
   

