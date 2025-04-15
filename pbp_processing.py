import pandas as pd
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import time

eventType = {
    "Jump Ball" : {
        "Home Won" : "Home Won Jump Ball",
        "Away Won" : "Away Won Jump Ball",
    },
    "2PT Missed" : "2PT Missed",
    "2PT Made" : "2PT Made",
    "3PT Missed" : "3PT Missed",
    "3PT Made" : "3PT Made",
    "Rebound" : {
        "Offensive" : "Offensive Rebound",
        "Defensive" : "Defensive Rebound",
    },
    "Turnover" : {
        "Bad Pass" : "Bad Pass Turnover",
        "Lost Ball" : "Lost Ball Turnover",
        "Traveling" : "Traveling Violation",
        "Out Of Bounds" : "Out Of Bounds",
        "3 Second Violation" : "Offensive 3 Second Violation",
        "Discontinue Dribble" : "Double Dribble",
        "Backcourt Turnover" : "Over And Back",
        "Double Dribble" : "Double Dribble",
        "Offensive Goaltending" : "Offensive Goaltending",
        "Lane Violation" : "Offensive Lane Violation",
        "Jump Ball Violation" : "Jump Ball Violation",
        "Illegal Pick" : "Illegal Screen",
        "Kicked Ball Violation" : "Offensive Kicked Ball Violation",
        "Illegal Assist Turnover" : "Illegal Assist Turnover",
        "Inbound Turnover" : "Inbound Turnover",
        "Shot Clock Turnover" : "Shot Clock Violation",
        "5 Second Violation" : "5 Second Violation",
        "8 Second Violation" : "8 Second Violation",
        "Isolation Violation" : "Isolation Violation",
        "Palming Turnover" : "Palming Turnover",
        " " : "General Turnover",
        "Post Up Turnover" : "Post Up Turnover",
        "Double Personal Turnover" : "Double Personal Turnover",
        "Poss Lost Ball Turnover" : "Lost Ball Turnover",
        "Out of Bounds Lost Ball Turnover" : "Lost Ball Turnover",
        "Step Out of Bounds Turnover" : "Out Of Bounds",
        "Stolen Pass Turnover" : "Stolen Pass Turnover",
        "Offensive Foul Turnover" : "Offensive Foul",
        "5 Second Inbound Turnover" : "5 Second Violation",
        "Illegal Screen Turnover" : "Illegal Screen",
        "Swinging Elbows Turnover" : "Swinging Elbows Turnover",
        "Basket from Below Turnover" : "Basket From Below Violation",
        "Punched Ball Turnover" : "Punched Ball Turnover",
        "Opposite Basket Turnover" : "Lost Ball Turnover",
        "Player Out of Bounds Violation Turnover" : "Player Out Of Bounds Violation",
        "Excess Timeout Turnover" : "Excess Timeout Turnover",
        "Too Many Players Turnover" : "Too Many Players Turnover",
        "Out of Bounds - Bad Pass Turnover" : "Bad Pass Turnover",
    },
    "Foul" : {
        "Shooting" : "Shooting Foul",
        "Personal" : "Personal Foul",
        "Double Technical" : "Double Technical Foul",
        "Offensive" : "Offensive Foul",
        "Loose Ball" : "Loose Ball Foul",
        "Flagrant Type 1" : "Flagrant 1 Foul",
        "Technical" : "Technical Foul",
        "Away From Play" : "Away From The Play Foul",
        "Clear Path" : "Clear Path Foul",
        "Hanging Technical" : "Hanging Technical Foul",
        "Defense 3 Second" : "Defensive 3 Second Violation",
        "Taunting Technical" : "Taunting Technical",
        "Inbound" : "Inbound Foul",
        "Double Personal" : "Double Personal Foul",
        "Flagrant Type 2" : "Flagrant 2 Foul",
        "Delay Technical" : "Delay Of Game Technical",
        "Punching" : "Punching Foul",
        "Elbow" : "Elbow Foul",
        "Non-Unsportsmanlike Technical" : "Unsportsmanlike Technical",
        "Offensive Charge" : "Charging Foul",
        "Personal Block" : "Blocking Foul",
        "Personal Take" : "Personal Foul",
        "Shooting Block" : "Shooting Foul",
        "Excess Timeout Technical" : "Excess Timeout Technical",
        "Too Many Players Technical" : "Too Many Players Technical",
        "Transition Take" : "Transition Take Foul",
    },
    "Violation" : {
        "Lane" : "Defensive Lane Violation",
        "Jump Ball" : "Jump Ball",
        "Defensive Goaltending" : "Defensive Goaltending",
        "Double Lane" : "Double Lane Violation",
        "Kicked Ball" : "Defensive Kicked Ball Violation",
        "Delay Of Game" : "Delay Of Game Violation",
        " " : "General Violation",
    },
    "Inbound" : "Inbound",
    "Free Throw" : {
        "Free Throw 1 of 2" : "Free Throw 1 of 2",
        "Free Throw 1 of 1" : "Free Throw 1 of 1",
        "Free Throw 1 of 3" : "Free Throw 1 of 3",
        "Free Throw 2 of 2" : "Free Throw 2 of 2",
        "Free Throw 2 of 3" : "Free Throw 2 of 3",
        "Free Throw 3 of 3" : "Free Throw 3 of 3",
        "Free Throw Clear Path" : "Clear Path Free Throw",
        "Free Throw Clear Path 1 of 2" : "Clear Path Free Throw 1 of 2",
        "Free Throw Clear Path 2 of 2" : "Clear Path Free Throw 2 of 2",
        "Free Throw Flagrant 1 of 2" : "Flagrant Foul Free Throw 1 of 2",
        "Free Throw Flagrant 2 of 2" : "Flagrant Foul Free Throw 2 of 2",
        "Free Throw Flagrant 1 of 1" : "Flagrant Foul Free Throw 1 of 1",
        "Free Throw Flagrant 1 of 3" : "Flagrant Foul Free Throw 1 of 3",
        "Free Throw Flagrant 2 of 3" : "Flagrant Foul Free Throw 2 of 3",
        "Free Throw Flagrant 3 of 3" : "Flagrant Foul Free Throw 3 of 3",
        "Free Throw Technical" : "Technical Foul Free Throw",
        "Free Throw Technical 1 of 2" : "Technical Foul Free Throw 1 of 2",
        "Free Throw Technical 2 of 2" : "Technical Foul Free Throw 2 of 2",
    },
    "Ejection" : {
        "Second Technical" : "Second Technical Ejection",
        "Second Flagrant Type 1" : "Second Flagrant 1 Ejection",
        "Other" : "General Ejection",
        "First Flagrant Type 2" : "Flagrant 2 Ejection",
        " " : "General Ejection"
    },
    "Timeout" : {
        "Short" : "20 Second Timeout",
        "Regular" : "Full Timeout",
        "Official TV" : "TV Timeout",
        "Official" : "Official Timeout",
        " " : "Official Timeout",
        "Advance" : "Advance Timeout",
        "Coach Challenge" : "Coach Challenge",
    },
    "period" : {
        "start" : "Start Of Quarter",
        "end" : "End of Quarter",
    },
    "Instant Replay" : {
        "Support Ruling" : "Instant Replay: Call Stands",
        "Ruling Stands" : "Instant Replay: Call Stands",
        "Overturn Ruling" : "Instant Replay: Call Overturned",
        "Altercation Ruling" : "Instant Replay: Altercation Ruling",
        "Coach Challenge Support Ruling" : "Coach Challenge: Call Stands",
        "Coach Challenge Ruling Stands" : "Coach Challenge: Call Stands",
        "Coach Challenge Overturn Ruling" : "Coach Challenge: Call Overturned",
        "Replay Center" : "Instant Replay: Replay Center",
    },
    "Block" : "Block",
    "Steal" : "Steal",
}

def get_home_away(df):
    def deduce_home_away(row, gameTeamsDF):
        scoringTeam = row['team']
        gameid = row['gameid']
        allTeams = gameTeamsDF[gameid]
        otherTeam = next(team for team in allTeams if team != scoringTeam)
    
        if row['h_pts'] > 0:
            return pd.Series({'homeTeam': scoringTeam, 'awayTeam': otherTeam})
        else:
            return pd.Series({'homeTeam': otherTeam, 'awayTeam': scoringTeam})
        
    gameTeamsDF = df[['gameid', 'team']].drop_duplicates().dropna(subset=['team']).groupby('gameid')['team'].apply(list).to_dict()
    
    scoringPlays = df[(df['h_pts'] > 0) | (df['a_pts'] > 0)].copy()
    scoringPlays = scoringPlays.sort_values(['gameid', 'Unnamed: 0'])
    firstScores = scoringPlays.groupby('gameid').first().reset_index()
    
    homeAway = firstScores.apply(lambda row: deduce_home_away(row, gameTeamsDF), axis=1)
    homeAway['gameid'] = firstScores['gameid']
    df = df.merge(homeAway, on='gameid', how='left')
    return df 

def set_ft_result(df):
    new_df = df.copy()
    def get_ft_result(row):
        if row["type"] == "Free Throw" and "PTS" in row["desc"]:
            return "Made"
        elif row["type"] == "Free Throw":
            return "Missed"
        else:
            return row["result"]
    new_df["result"] = df.apply(get_ft_result, axis=1)
    return new_df

def set_jumpball_subtype(df):
    new_df = df.copy()
    def get_jumpball_subtype(row):
        if row["type"] == "Jump Ball":
            return "Home Won" if row["result"] == row["homeTeam"] else "Away Won"
        else:
            return row["subtype"]
    new_df["subtype"] = new_df.apply(get_jumpball_subtype, axis=1)
    return new_df
    
def add_inbounds(df):
    is_made_shot = df["type"] == "Made Shot"
    is_turnover = df["type"] == "Turnover"
    not_followed_by_steal = ~(
        (df["next_type"] == "Steal") &
        (df["next_team"] != df["team"]) &
        (df["gameid"] == df["next_gameid"])
    )
    is_dead_ball_turnover = is_turnover & not_followed_by_steal
    is_final_free_throw = (
        (df["type"] == "Free Throw") &
        (df["subtype"].isin([
            "Free Throw 2 of 2",
            "Free Throw 3 of 3",
            "Free Throw 1 of 1",
            "Free Throw Flagrant 3 of 3",
            "Free Throw Flagrant 2 of 2",
            "Free Throw Clear Path 2 of 2",
            "Free Throw Technical 2 of 2",
        ])) & (df["result"] == "Made")
    )
    is_timeout = df["type"] == "Timeout"
    is_deadball_foul = ((df["type"] == "Foul") & ((df["subtype"] == "Offensive") | (df["subtype"] == "Personal")))
    is_and1 = (
        (df["next_type"] == "Foul") &
        (df["next_subtype"] == "Shooting") &
        (df["next2_subtype"] == "Free Throw 1 of 1")
    )
    timeout_then_foul = (
        (df["next_type"] == "Timeout") &
        (df["next2_type"] == "Foul") &
        (df["next2_team"] != df["team"]) &
        (df["next2_subtype"] == "Shooting")
    )
    is_and1_case = is_and1 | timeout_then_foul
    made_shot_followed_by_timeout = (
        (df["next_type"] == "Timeout") &
        (df["next_gameid"] == df["gameid"])
    )
    valid_made_shot = (
        is_made_shot &
        ~is_and1_case &
        ~made_shot_followed_by_timeout
    )
    timeout_after_shooting_foul = (
        (df["type"] == "Timeout") &
        ((df["type"].shift(1) == "Foul") & (df["subtype"].shift(1) == "Shooting"))
    )
    valid_timeout = is_timeout & ~timeout_after_shooting_foul
    ends_with_inbound = is_dead_ball_turnover | is_final_free_throw | valid_timeout | is_deadball_foul | valid_made_shot
    
    inbound_rows = df[ends_with_inbound].copy()
    
    inbound_rows["type"] = "Inbound"
    inbound_rows["team"] = pd.NA
    inbound_rows["subtype"] = pd.NA
    inbound_rows["player"] = pd.NA
    inbound_rows["EventIndex"] = inbound_rows["EventIndex"] + 0.1  # So they sort after the original event
    inbound_rows["desc"] = "Inbound pass"
    
    df_with_inbounds = pd.concat([df, inbound_rows], ignore_index=True)
    df = df_with_inbounds.sort_values(by=["gameid", "EventIndex"]).reset_index(drop=True)
    return df

def set_inbound_team(df):
    new_df = df.copy()
    def get_inbound_team(row, past_team):
        previous_team = past_team.loc[row.name]
        return row["homeTeam"] if previous_team != row["homeTeam"] else row["awayTeam"]
    past_team = new_df["team"].shift(1)
    new_df.loc[new_df["type"] == "Inbound", "team"] = new_df[new_df["type"] == "Inbound"].apply(lambda row: get_inbound_team(row, past_team), axis=1)
    return new_df
 
def set_shot_type(df):
    new_df = df.copy()
    def get_shot_type(row):
        if isinstance(row["desc"], str) and isinstance(row["type"], str):
            if "3PT" in row["desc"] and row["type"] == "Made Shot":
                return "3PT Made"
            elif "3PT" in row["desc"] and row["type"] == "Missed Shot":
                return "3PT Missed"
            elif row["type"] == "Made Shot":
                return "2PT Made"
            elif row["type"] == "Missed Shot":
                return "2PT Missed"
            else:
                return row["type"]
        else:
            return row["type"]
    new_df["type"] = new_df.apply(get_shot_type, axis=1)
    return new_df
    
def fix_free_throw_sequences(df):
    df = df.sort_values(by=["gameid", "EventIndex"]).reset_index(drop=True)
    df["row_idx"] = df.index

    # Identify Free Throw 1 of 2 and 2 of 2 rows
    ft1s = df[(df["type"] == "Free Throw") & (df["subtype"] == "Free Throw 1 of 2")][["gameid", "team", "row_idx", "EventIndex"]].copy()
    ft2s = df[(df["type"] == "Free Throw") & (df["subtype"] == "Free Throw 2 of 2")][["gameid", "team", "row_idx"]].copy()

    # Rename for merge clarity
    ft1s = ft1s.rename(columns={"row_idx": "ft1_idx", "EventIndex": "ft1_EventIndex"})
    ft2s = ft2s.rename(columns={"row_idx": "ft2_idx"})

    # Use merge_asof to find the first FT2 after FT1 (within same game + team)
    ft1_ft2 = pd.merge_asof(
        ft1s.sort_values("ft1_idx"),
        ft2s.sort_values("ft2_idx"),
        by=["gameid", "team"],
        left_on="ft1_idx",
        right_on="ft2_idx",
        direction="forward"
    )

    # Drop rows in between each FT1 and its matching FT2
    to_drop = (
        ft1_ft2[ft1_ft2["ft2_idx"].notna()]
        .apply(lambda row: list(range(int(row["ft1_idx"]) + 1, int(row["ft2_idx"]))), axis=1)
        .explode()
        .dropna()
        .astype(int)
        .tolist()
    )

    # Also drop the original FT2
    to_drop += ft1_ft2["ft2_idx"].dropna().astype(int).tolist()

    # Prepare FT2s to be reinserted right after FT1s
    new_ft2_rows = df.loc[ft1_ft2["ft2_idx"].dropna().astype(int)].copy().reset_index(drop=True)
    new_ft2_rows["EventIndex"] = ft1_ft2["ft1_EventIndex"].reset_index(drop=True) + 0.01

    # Drop the bad rows and insert the fixed FT2s
    df = df.drop(index=to_drop).reset_index(drop=True)
    df = pd.concat([df, new_ft2_rows], ignore_index=True)
    df = df.sort_values(by=["gameid", "EventIndex"]).reset_index(drop=True)

    return df.drop(columns="row_idx")

def set_event_type(df):
    new_df = df.copy()
    def get_event_type(row):
        if row["type"] in eventType and isinstance(eventType[row["type"]], str):
            return eventType[row["type"]]
        elif row["type"] in eventType and row["subtype"] in eventType[row["type"]]:
            return eventType[row["type"]][row["subtype"]]
        else:
            print(row["type"], row["subtype"])
    new_df = new_df[~((new_df["type"].isna()) & (new_df["subtype"].isna()))]
    new_df["eventType"] = new_df.apply(get_event_type, axis=1)
    return new_df
     
def update_blocks_steals_charges(df):
    new_df = df.copy()
    
    blocked_2PT = ((new_df["next_type"] == "Block") & (new_df["type"] == "2PT Missed"))
    blocked_3PT = ((new_df["next_type"] == "Block") & (new_df["type"] == "3PT Missed"))
    block_rows = new_df[new_df["type"] == "Block"].index
    new_df.loc[blocked_2PT, ["type", "eventType"]] = ["2PT Blocked", "2PT Blocked"]
    new_df.loc[blocked_3PT, ["type", "eventType"]] = ["3PT Blocked", "3PT Blocked"]
    new_df = new_df.drop(block_rows, axis=0)
    
    steal_turnover_rows = new_df[new_df["next_type"] == "Steal"].index
    new_df = new_df.drop(steal_turnover_rows, axis=0)
    
    charging_foul_rows = new_df[new_df["eventType"] == "Offensive Foul"].index
    new_df = new_df.drop(charging_foul_rows, axis=0)
    
    return new_df

def set_possession_change(df):
    new_df = df.copy()
    new_df["possessionChange"] = pd.NA
    
    is_transition_take = new_df["prev2_eventType"] == "Transition Take Foul"
    is_final_free_throw = new_df["prev_eventType"].isin(["Free Throw 2 of 2", "Free Throw 1 of 1", 
                                                     "Free Throw 3 of 3", "Flagrant Foul Free Throw 3 of 3",
                                                     "Flagrant Foul Free Throw 2 of 2"])
    is_turnover = new_df["prev_type"] == "Turnover"
    is_offensive_foul = new_df["prev_eventType"] == "Charging Foul"
    is_steal = new_df["eventType"] == "Steal"
    is_defensive_rebound = new_df["eventType"] == "Defensive Rebound"
    is_possession_ending_make = ((new_df["prev_type"] == "2PT Made") | (new_df["prev_type"] == "3PT Made"))
    is_make_followed_by_timeout = (((new_df["prev2_type"] == "2PT Made") | (new_df["prev2_type"] == "3PT Made")) & (df["prev_type"] == "Timeout"))
    is_inbound = new_df["type"] == "Inbound"
    
    is_possession_change_inbound = (is_possession_ending_make | (is_final_free_throw & ~is_transition_take) | 
                                    is_turnover | is_offensive_foul | is_make_followed_by_timeout) & is_inbound
    
    is_possession_change = is_possession_change_inbound | is_defensive_rebound | is_steal
    
    new_df.loc[is_possession_change, "possessionChange"] = True
    new_df.loc[~is_possession_change, "possessionChange"] = False
    return new_df
    
def set_game_possession_team(game_id, game_df):
    possession_col = pd.Series(pd.NA, index=game_df.index, dtype="object")

    last_team = None
    possession_change = (game_df["possessionChange"] == True) & (game_df["type"] != "Jump Ball")

    for i in game_df.index:
        event = game_df.at[i, "eventType"]
        if event == "Home Won Jump Ball":
            last_team = game_df.at[i, "homeTeam"]
        elif event == "Away Won Jump Ball":
            last_team = game_df.at[i, "awayTeam"]
        elif possession_change[i] and last_team:
            last_team = (
                game_df.at[i, "awayTeam"]
                if last_team == game_df.at[i, "homeTeam"]
                else game_df.at[i, "homeTeam"]
            )
        if last_team:
            possession_col.at[i] = last_team
    return possession_col

def set_possession_team(df):
    new_df = df.copy()
    new_df["possessionTeam"] = pd.NA

    grouped = list(new_df.groupby("gameid"))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(set_game_possession_team, gid, gdf): gid for gid, gdf in grouped}
        for future in as_completed(futures):
            possession_result = future.result()
            new_df.loc[possession_result.index, "possessionTeam"] = possession_result
            
    return new_df

def set_possession_id(df):
    new_df = df.copy()
    new_df["possessionId"] = (
        (new_df["possessionChange"]) | (new_df["gameid"] != new_df["gameid"].shift(1))
    ).cumsum()
    return new_df

def process_season_pbp(season):
    df = pd.read_csv(f"data/pbp_{season}.csv")
    
    df["type"] = df["type"].str.strip()
    
    df = set_jumpball_subtype(
        set_ft_result(
        get_home_away(
            df[(df["type"] != "Substitution") & (df["subtype"] != "Foul")]
                      ).rename({"Unnamed: 0" : "EventIndex"}, axis=1)
        )).sort_values(by=["gameid", "EventIndex"]).reset_index(drop=True)
    
    # Shifted windows to check what's next
    df["next_type"] = df["type"].shift(-1)
    df["next_team"] = df["team"].shift(-1)
    df["next_subtype"] = df["subtype"].shift(-1)
    df["next2_subtype"] = df["subtype"].shift(-2)
    df["next2_type"] = df["type"].shift(-2)
    df["next2_team"] = df["team"].shift(-2)
    df["next_gameid"] = df["gameid"].shift(-1)
    
    df = update_blocks_steals_charges(
        set_event_type(
        fix_free_throw_sequences(
        set_shot_type(
        set_inbound_team(
        add_inbounds(df)
        ))))) 
    
    df["prev_eventType"] = df["eventType"].shift(1)
    df["prev2_eventType"] = df["eventType"].shift(2)
    df["prev_type"] = df["type"].shift(1)
    df["prev2_type"] = df["type"].shift(2)
    
    df = set_possession_id(
        set_possession_team(
            set_possession_change(df)
        )) [["EventIndex", "gameid", "possessionId", "period", "h_pts", "a_pts", "eventType", "x", "y", "dist", "season", 
             "minutes_remaining", "seconds_remaining", "homeTeam", "awayTeam", "possessionChange", "possessionTeam"]]

    df = df.rename({
        "season" : "season",
        "EventIndex" : "eventIndex",
        "gameid" : "gameId",
        "possessionId" : "possessionId", 
        "homeTeam" : "homeTeam",
        "awayTeam" : "awayTeam",
        "possessionTeam" : "possessionTeam",
        "possessionChange" : "possessionChange",
        "period" : "quarter",
        "minutes_remaining" : "minutesRemaining",
        "seconds_remaining" : "secondsRemaining",
        "eventType" : "eventType",
        "h_pts" : "homePoints",
        "a_pts" : "awayPoints",
        "x" : "X",
        "y" : "Y",
        "dist" : "shotDistance"
        })
    
    df.to_csv(f"data/pbp_{season}.csv")
    
def parallel_main():
    seasons = list(range(1997, 2024))
    with ProcessPoolExecutor(max_workers=27) as executor:
        futures = [executor.submit(process_season_pbp, season) for season in seasons]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Seasons"):
            pass

if __name__ == "__main__":
    start_time = time.time()
    parallel_main()
    print(time.time() - start_time)
