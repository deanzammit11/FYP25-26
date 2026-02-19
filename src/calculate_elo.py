import pandas as pd
from pathlib import Path

def add_elo_features(df, k_factor=20): # Adds Elo ratings and sets the intial elo rating values for 2023 and adds dynamic HFA values to the input dataframe
    df = df.copy() # The dataframe is copied and stored
    df["DateParsed"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce") # The match Date column is converted into datetime
    df["DateKey"] = df["DateParsed"].dt.normalize() # Each time in DateParsed is set to midnight for consistency
    elo_dir = Path("data/raw/Elo Ratings") # Stores the path which points to the directory where the Elo History of each club is stored
    elo_frames = [] # An empty list which will later store the elo history in a dataframe for each club is initialised
    for path in elo_dir.glob("*.csv"): # For each csv file in the elo directory
        elo_team = pd.read_csv(path) # The csv file is read
        elo_team["From"] = pd.to_datetime(elo_team["From"], errors="coerce") # The From column is converted into datetime
        elo_team["To"] = pd.to_datetime(elo_team["To"], errors="coerce") # The To column is converted into datetime
        elo_team["Level"] = pd.to_numeric(elo_team["Level"], errors="coerce") # The Level column is converted into numeric
        elo_team["Team"] = elo_team["Club"] # Ensures that the team names match
        elo_frames.append(elo_team[["Team", "Level", "Elo", "From", "To"]]) # The columns to keep are selected and the elo history for that club is added to elo_frames

    if elo_frames: # If at least on elo file was read
        elo_history = pd.concat(elo_frames, ignore_index=True) # Each seperate club dataframe is concatenated to form a single dataframe and the index is reset
        elo_history = elo_history.dropna(subset=["Team", "Elo", "From", "To"]).copy() # The rows where any of "Team", "Elo", "From", "To" columns is null are dropped and the new dataframe is copied
        elo_history = elo_history.sort_values(["Team", "From"]) # Rows are first sorted by team name and they are then sorted by From Date
    else: # If no elo file was read menaing elo_frames is empty
        raise FileNotFoundError("No Elo rating CSV files found in data/raw/Elo Ratings") # A file not found error is raised

    home_side = df[["HomeTeam", "Season", "DateParsed"]].rename(columns={"HomeTeam": "Team"}) # A temporary table to store the elo rating for the home side consisting of HomeTeam, Season and DateParsed columns with the HomeTeam column being renamed to Team is created
    home_side["Elo"] = pd.NA # The Elo column is added and assigned nulls as a temporary value
    home_2019_22 = home_side["Season"].between(2019, 2022) # The rows where the season is between 2019 and 2022 inclusive are stored
    if home_2019_22.any(): # If there are matches which fall between the 2019-2022 seasons
        home_side_2019_22 = home_side[home_2019_22 & home_side["DateParsed"].notna()] # The dataframe is filtered for rows which fall between the 2019-2022 seasons and the DateParsed is not null
        for team, group in home_side_2019_22.groupby("Team"): # For each team and group containing all matches for that team
            team_elo = elo_history[elo_history["Team"] == team] # The elo dataframe is filtered and only the rows for the respective club are stored
            if team_elo.empty: # If there is no Elo history
                continue # Jump to the next step in the for loop
            group_sorted = group.drop(columns="Elo").sort_values("DateParsed") # The Elo column is dropped and rows are sorted by match date
            team_elo_sorted = team_elo.sort_values("To") # Elo records are sorted by their end date
            merged = pd.merge_asof(group_sorted, team_elo_sorted, left_on="DateParsed", right_on="To", direction="backward") # Compares DateParsed from group_sorted to to from team_elo_sorted and the largest/closest To value which is ≤ to the match date is picked and those 2 rows are merged
            home_side.loc[group_sorted.index, "Elo"] = merged["Elo"].values # The Elo ratings of merged are assigned to home_side based on the matching index since group_sorted is a subset of home_side
    away_side = df[["AwayTeam", "Season", "DateParsed"]].rename(columns={"AwayTeam": "Team"}) # A temporary table to store the elo rating for the away side consisting of AwayTeam, Season and DateParsed columns with the AwayTeam column being renamed to Team is created
    away_side["Elo"] = pd.NA # The Elo column is added and assigned nulls as a temporary value
    away_2019_22 = away_side["Season"].between(2019, 2022) # The rows where the season is between 2019 and 2022 inclusive are stored
    if away_2019_22.any(): # If there are matches which fall between the 2019-2022 seasons
        away_side_2019_22 = away_side[away_2019_22 & away_side["DateParsed"].notna()] # The dataframe is filtered for rows which fall between the 2019-2022 seasons and the DateParsed is not null
        for team, group in away_side_2019_22.groupby("Team"): # For each team and group containing all matches for that team
            team_elo = elo_history[elo_history["Team"] == team] # The elo dataframe is filtered and only the rows for the respective club are stored
            if team_elo.empty: # If there is no Elo history
                continue # Jump to the next step in the for loop
            group_sorted = group.drop(columns="Elo").sort_values("DateParsed") # The Elo column is dropped and rows are sorted by match date
            team_elo_sorted = team_elo.sort_values("To") # Elo records are sorted by their end date
            merged = pd.merge_asof(group_sorted, team_elo_sorted, left_on="DateParsed", right_on="To", direction="backward") # Compares DateParsed from group_sorted to to from team_elo_sorted and the largest/closest To value which is ≤ to the match date is picked and those 2 rows are merged
            away_side.loc[group_sorted.index, "Elo"] = merged["Elo"].values # The Elo ratings of merged are assigned to away_side based on the matching index since group_sorted is a subset of away_side
    premier_league_all = pd.read_csv("data/processed/eng1_all_seasons.csv") # The premier league dataframe is read from the specified directory
    premier_league_all = premier_league_all[["Season", "Date", "HomeTeam", "AwayTeam", "FullTimeResult"]].copy() # Only the columns needed for Elo and HFA updates are kept
    premier_league_all["Season"] = pd.to_numeric(premier_league_all["Season"], errors="coerce") # The Season column is converted into numeric
    premier_league_all = premier_league_all[premier_league_all["Season"] >= 2019] # Only rows where the season is 2019 or later are kept
    premier_league_all["DateParsed"] = pd.to_datetime(premier_league_all["Date"], dayfirst=True, errors="coerce") # The Date column is converted into datetime
    premier_league_all["DateKey"] = premier_league_all["DateParsed"].dt.normalize() # Each time in DateParsed is set to midnight for consistency
    premier_league_all = premier_league_all.rename(columns={"FullTimeResult": "ResultCode"}) # FullTimeResult is renamed to ResultCode to align with championship naming
    premier_league_all = premier_league_all.dropna(subset=["DateParsed", "HomeTeam", "AwayTeam", "ResultCode"]) # Rows with missing date, team names or result are dropped

    championship_all = pd.read_csv("data/raw/Championship 2019-2023.csv") # The championship dataframe is read from the specified directory
    championship_all = championship_all[["Season", "Date", "HomeTeam", "AwayTeam", "FT Result"]].copy() # Only the columns needed for HFA updates are kept
    championship_all["Season"] = pd.to_numeric(championship_all["Season"], errors="coerce") # The Season column is converted into numeric
    championship_all = championship_all[championship_all["Season"] >= 2019] # Only rows where the season is 2019 or later are kept
    championship_all["DateParsed"] = pd.to_datetime(championship_all["Date"], dayfirst=True, errors="coerce") # The Date column is converted into datetime
    championship_all["DateKey"] = championship_all["DateParsed"].dt.normalize() # Each time in DateParsed is set to midnight for consistency
    championship_all = championship_all.rename(columns={"FT Result": "ResultCode"}) # FT Result is renamed to ResultCode to align with premier league naming
    championship_all = championship_all.dropna(subset=["DateParsed", "HomeTeam", "AwayTeam", "ResultCode"]) # Rows with missing date, team names or result are dropped

    initial_map_source = pd.concat([
            premier_league_all[["HomeTeam", "AwayTeam", "DateParsed"]], # The HomeTeam, AwayTeam, and DateParsed columns are selected from the premier league dataframe
            championship_all[["HomeTeam", "AwayTeam", "DateParsed"]], # The HomeTeam, AwayTeam, and DateParsed columns are selected from the championship dataframe
        ], ignore_index=True) # The selected columns from the premier league fixtures dataframe and championship fixtures dataframe are concatenated to form a single dataframe and the index is reset
    initial_elo_map = {} # An empty dictionary which will later store the initial elo ratings used for the combined fixtures is initialised
    season_teams = pd.concat([ # Home and away team appearances are stacked into one dataframe and the team column name is standardised
            initial_map_source[["HomeTeam", "DateParsed"]].rename(columns={"HomeTeam": "Team"}), # HomeTeam and DateParsed columns are stored in a new dataframe with the HomeTeam column being renamed to Team
            initial_map_source[["AwayTeam", "DateParsed"]].rename(columns={"AwayTeam": "Team"}), # AwayTeam and DateParsed columns are stored in a new dataframe with the AwayTeam column being renamed to Team
        ], ignore_index=True) # The above 2 dataframes are concatenated and resets the index with the resulting dataframe containing the appareance of each team along with the match date
    first_match_dates = season_teams.groupby("Team")["DateParsed"].min() # The earliest match date for each team is selected
    for team, first_date in first_match_dates.items(): # For each team and the corresponding first match date
        team_elo = elo_history[elo_history["Team"] == team] # The elo history for that team is selected
        if team_elo.empty or pd.isna(first_date): # If the team has no Elo history or the first date is null
            continue # Jump to the next team
        prior_elo = team_elo[team_elo["To"] <= first_date] # Only Elo records up to and including the first match date are kept
        if prior_elo.empty: # If there are no Elo records before that first match date
            continue # Jump to the next team
        closest_idx = prior_elo["To"].idxmax() # The index of the most recent Elo record before the first match date is selected
        initial_elo_map[team] = prior_elo.loc[closest_idx, "Elo"] # The corresponding Elo rating is stored as the initial Elo for that team

    home_2023 = home_side["Season"] == 2023 # The dataframe is filterted only storing rows where the Season is 2023
    if home_2023.any(): # If there are matches which fall under the 2023 season
        home_side.loc[home_2023, "Elo"] = home_side.loc[home_2023, "Team"].map(initial_elo_map) # For the home side for every fixture falling under the 2023 season the initial Elo rating from initial_elo_map is looked up and that value is written into the Elo column setting the Elo rating for each fixture in 2023 to the initial elo rating

    away_2023 = away_side["Season"] == 2023 # The dataframe is filterted only storing row where the Season is 2023
    if away_2023.any(): # If there are matches which fall under the 2023 season
        away_side.loc[away_2023, "Elo"] = away_side.loc[away_2023, "Team"].map(initial_elo_map) # For the away side for every fixture falling under the 2023 season the initial Elo rating from initial_elo_map is looked up and that value is written into the Elo column setting the Elo rating for each fixture in 2023 to the initial elo rating

    df["HomeElo"] = home_side["Elo"].values # The Elo ratings of the home side before each fixture with 2023 all set to initial ratings are added to the main dataframe
    df["AwayElo"] = away_side["Elo"].values # The Elo ratings of the away side before each fixture with 2023 all set to initial ratings are added to the main dataframe

    dynamic_hfa = 66.7 # The initial dynamic HFA value is set
    day_hfa_map = {} # A dictionary mapping each date to the HFA value used at the start of that day is initialised
    elo_ratings = initial_elo_map.copy() # The Elo ratings map which will be used for the simulation is initialised starting with a copy of the initial ratings
    premier_league_dates = set(premier_league_all["DateKey"].dropna().unique().tolist()) # All unique premier league fixture dates are stored in a list after removing null values
    championship_dates = set(championship_all["DateKey"].dropna().unique().tolist()) # All unique championship fixture dates are stored in a list after removing null values
    all_dates = sorted(premier_league_dates.union(championship_dates)) # Both date sets are combined and are sorted in chronological order
    for match_date in all_dates: # For each date where either league has a fixture
        day_hfa_map[match_date] = dynamic_hfa # Stores the HFA value to be used for fixtures on that date
        day_delta_sum = 0.0 # The delta sum for that day is set to 0 at the beginning of each day iteration

        premier_league_matches_today = premier_league_all[premier_league_all["DateKey"] == match_date].sort_values(["HomeTeam", "AwayTeam"]) # The premier league fixtures for that date are selected and are sorted by home team and then away team names
        for _, fixture in premier_league_matches_today.iterrows(): # Iterates over each premier league fixture on that date
            home_team = fixture["HomeTeam"] # The home team is stored
            away_team = fixture["AwayTeam"] # The away team is stored
            if home_team not in elo_ratings: # If the home team is not found in the Elo ratings map
                raise KeyError(f"Missing Elo rating entry for home team: {home_team} on {fixture['Date']} vs {away_team}") # A key error is raised
            if away_team not in elo_ratings: # If the away team is not found in the Elo ratings map
                raise KeyError(f"Missing Elo rating entry for away team: {away_team} on {fixture['Date']} vs {home_team}") # A key error is raised
            home_rating = elo_ratings[home_team] # The current Elo rating for the home team is stored
            away_rating = elo_ratings[away_team] # The current Elo rating for the away team is stored
            if pd.isna(home_rating): # If the home Elo rating is missing
                raise ValueError(f"Missing Elo rating value for home team: {home_team} on {fixture['Date']} vs {away_team}") # A value error is raised
            if pd.isna(away_rating): # If the away Elo rating is missing
                raise ValueError(f"Missing Elo rating value for away team: {away_team} on {fixture['Date']} vs {home_team}") # A value error is raised

            if fixture["ResultCode"] == "H": # If the home side won
                score_home = 1 # The score result is 1
            elif fixture["ResultCode"] == "D": # If the home side drew
                score_home = 0.5 # The score result is 0.5
            else: # If the home side lost
                score_home = 0 # The score result is 0
            home_rating_adjusted = home_rating + dynamic_hfa # The elo rating for the home side is adjusted to include the home field advantage
            expected_home = 1 / (1 + 10 ** (-(home_rating_adjusted - away_rating) / 400)) # The home win probability is computed
            delta_home = k_factor * (score_home - expected_home) # The Elo change for the home side is calculated
            elo_ratings[home_team] = home_rating + delta_home # The new Elo rating for the home side is calculated
            elo_ratings[away_team] = away_rating - delta_home # The new Elo rating for the away side is calculated
            day_delta_sum += delta_home # The home Elo change is added to the daily delta sum

        championship_matches_today = championship_all[championship_all["DateKey"] == match_date].sort_values(["HomeTeam", "AwayTeam"]) # The championship fixtures for that date are selected and are sorted by home team and then away team names
        for _, fixture in championship_matches_today.iterrows(): # Iterates over each championship fixture on that date
            home_team = fixture["HomeTeam"] # The home team is stored
            away_team = fixture["AwayTeam"] # The away team is stored
            if home_team not in elo_ratings: # If the home team is not found in the Elo ratings map
                raise KeyError(f"Missing Elo rating entry for home team: {home_team} on {fixture['Date']} vs {away_team}") # A key error is raised
            if away_team not in elo_ratings: # If the away team is not found in the Elo ratings map
                raise KeyError(f"Missing Elo rating entry for away team: {away_team} on {fixture['Date']} vs {home_team}") # A key error is raised
            home_rating = elo_ratings[home_team] # The current Elo rating for the home team is stored
            away_rating = elo_ratings[away_team] # The current Elo rating for the away team is stored
            if pd.isna(home_rating): # If the home Elo rating is missing
                raise ValueError(f"Missing Elo rating value for home team: {home_team} on {fixture['Date']} vs {away_team}") # A value error is raised
            if pd.isna(away_rating): # If the away Elo rating is missing
                raise ValueError(f"Missing Elo rating value for away team: {away_team} on {fixture['Date']} vs {home_team}") # A value error is raised

            if fixture["ResultCode"] == "H": # If the home side won
                score_home = 1 # The score result is 1
            elif fixture["ResultCode"] == "D": # If the home side drew
                score_home = 0.5 # The score result is 0.5
            else: # If the home side lost
                score_home = 0 # The score result is 0
            home_rating_adjusted = home_rating + dynamic_hfa # The elo rating for the home side is adjusted to include the home field advantage
            expected_home = 1 / (1 + 10 ** (-(home_rating_adjusted - away_rating) / 400)) # The home win probability is computed
            delta_home = k_factor * (score_home - expected_home) # The Elo change for the home side is calculated
            elo_ratings[home_team] = home_rating + delta_home # The new Elo rating for the home side is calculated
            elo_ratings[away_team] = away_rating - delta_home # The new Elo rating for the away side is calculated
            day_delta_sum += delta_home # The home Elo change is added to the daily delta sum

        dynamic_hfa += day_delta_sum * 0.075 # The dynamic HFA value is updated at the end of the day using the daily home delta sum

    df["HFA"] = df["DateKey"].map(day_hfa_map) # Each fixture is assigned the HFA value based on the day to HFA map
    if df["HFA"].isna().any(): # If the HFA value of any fixture is null
        missing_dates = sorted(df.loc[df["HFA"].isna(), "Date"].dropna().unique().tolist()) # The unique fixture dates where HFA is null are stored in a list
        raise ValueError(f"Missing HFA mapping for fixture dates: {missing_dates}") # A value error is raised

    df["EloTierHome"] = 4 # Sets the Elo tier for the home side in all fixtures to 4
    df.loc[df["HomeElo"] >= 1700, "EloTierHome"] = 3 # If the Elo rating for the home side before that fixture is greater than or equal to 1700 the tier is set to 3
    df.loc[df["HomeElo"] >= 1800, "EloTierHome"] = 2 # If the Elo rating for the home side before that fixture is greater than or equal to 1800 the tier is set to 2
    df.loc[df["HomeElo"] >= 1900, "EloTierHome"] = 1 # If the Elo rating for the home side before that fixture is greater than or equal to 1900 the tier is set to 1
    df["EloTierAway"] = 4 # Sets the Elo tier for the away side in all fixtures to 4
    df.loc[df["AwayElo"] >= 1700, "EloTierAway"] = 3 # If the Elo rating for the away side before that fixture is greater than or equal to 1700 the tier is set to 3
    df.loc[df["AwayElo"] >= 1800, "EloTierAway"] = 2 # If the Elo rating for the away side before that fixture is greater than or equal to 1800 the tier is set to 2
    df.loc[df["AwayElo"] >= 1900, "EloTierAway"] = 1 # If the Elo rating for the away side before that fixture is greater than or equal to 1900 the tier is set to 1
    if "DateKey" in df.columns: # If the DateKey column is present
        df = df.drop(columns=["DateKey"]) # The DateKey column is removed from dataframe df
    return df # The final dataframe with the added elo features is returned

def predict_2023_with_elo_updates(model, test_df, feature_columns, prediction_to_result=None, k_factor=20):
    test_df = test_df.copy() # The test dataframe is copied and stored
    test_df["DateParsed"] = pd.to_datetime(test_df["Date"], dayfirst=True, errors="coerce") # The Date column is converted into datetime
    test_df["Season"] = pd.to_numeric(test_df["Season"], errors="coerce") # The Season column is converted into numeric
    test_df = test_df[test_df["Season"] == 2023].copy() # The dataframe is filtered only keeping rows where Season is 2023

    required_feature_columns = {"HomeElo", "AwayElo", "HFA"} # The required precomputed Elo and HFA columns are defined
    if not required_feature_columns.issubset(test_df.columns): # If any required precomputed column is missing
        missing = required_feature_columns.difference(test_df.columns) # The missing columns are identified
        raise ValueError(f"Missing required columns in test_df: {sorted(missing)}") # A value error is raised

    season_sorted = test_df.sort_values(["DateParsed", "HomeTeam", "AwayTeam"]) # Rows are first sorted in chronological order by date and they are then sorted by HomeTeam and AwayTeam
    season_sorted["DateKey"] = season_sorted["DateParsed"].dt.normalize() # Each time in DateParsed is set to midnight for consistency
    predictions_by_index = {} # An empty dictionary which will later store the predicted class for each row index is initialised
    current_ratings = {} # An empty dictionary which will store the latest Elo rating for each team is initialised
    if season_sorted["HFA"].isna().all(): # If all HFA values are missing for 2023 fixtures
        raise ValueError("Missing HFA values for all 2023 fixtures in test_df") # A value error is raised
    current_hfa = season_sorted["HFA"].dropna().iloc[0] # The initial HFA value is set to the value of the first 2023 fixture

    championship_2023 = pd.read_csv("data/raw/Championship 2019-2023.csv") # The championship dataframe is read from the specified directory
    championship_2023 = championship_2023[["Season", "Date", "HomeTeam", "AwayTeam", "FT Result"]].copy() # Only the columns needed for HFA updates are kept
    championship_2023["Season"] = pd.to_numeric(championship_2023["Season"], errors="coerce") # The Season column is converted into numeric
    championship_2023 = championship_2023[championship_2023["Season"] == 2023] # Only rows where season is 2023 are kept
    championship_2023["DateParsed"] = pd.to_datetime(championship_2023["Date"], dayfirst=True, errors="coerce") # The Date column is converted into datetime
    championship_2023["DateKey"] = championship_2023["DateParsed"].dt.normalize() # Each time in DateParsed is set to midnight for consistency
    championship_2023 = championship_2023.rename(columns={"FT Result": "ResultCode"}) # FT Result is renamed to ResultCode to align with premier league naming
    championship_2023 = championship_2023.dropna(subset=["DateParsed", "HomeTeam", "AwayTeam", "ResultCode"]) # Rows with missing date, teams or result are dropped

    elo_frames = [] # An empty list which will later store the elo history in a dataframe for each club is initialised
    for path in Path("data/raw/Elo Ratings").glob("*.csv"): # For each csv file in the elo directory
        elo_team = pd.read_csv(path) # The csv file is read
        elo_team["From"] = pd.to_datetime(elo_team["From"], errors="coerce") # The From column is converted into datetime
        elo_team["To"] = pd.to_datetime(elo_team["To"], errors="coerce") # The To column is converted into datetime
        elo_team["Level"] = pd.to_numeric(elo_team["Level"], errors="coerce") # The Level column is converted into numeric
        elo_team["Team"] = elo_team["Club"] # Ensures that the team names match
        elo_frames.append(elo_team[["Team", "Level", "Elo", "From", "To"]]) # The columns to keep are selected and the elo history for that club is added to elo_frames
    if elo_frames: # If at least one elo file was read
        elo_history = pd.concat(elo_frames, ignore_index=True) # Each seperate club dataframe is concatenated to form a single dataframe and the index is reset
        elo_history = elo_history.dropna(subset=["Team", "Elo", "From", "To"]).copy() # The rows where any of Team, Elo, From, To is null are dropped and the new dataframe is copied
        elo_history = elo_history.sort_values(["Team", "From"]) # Rows are first sorted by team name and then by From Date
    else: # If no elo file was read meaning elo_frames is empty
        raise FileNotFoundError("No Elo rating CSV files found in data/raw/Elo Ratings") # A file not found error is raised

    premier_league_dates = set(season_sorted["DateKey"].dropna().unique().tolist()) # All unique premier league fixture dates are stored in a list after removing null values
    championship_dates = set(championship_2023["DateKey"].dropna().unique().tolist()) # All unique championship fixture dates are stored in a list after removing null values
    all_dates = sorted(premier_league_dates.union(championship_dates)) # Both date sets are combined and are sorted in chronological order

    for match_date in all_dates: # For each date where either league has a fixture
        day_delta_sum = 0.0 # The delta sum for that day is set to 0 at the beginning of each day iteration

        premier_league_matches_today = season_sorted[season_sorted["DateKey"] == match_date].sort_values(["HomeTeam", "AwayTeam"]) # The premier league fixtures for that date are selected and are sorted by home team and then away team names
        for idx, fixture in premier_league_matches_today.iterrows(): # Iterates over each premier league fixture on that date
            home_team = fixture["HomeTeam"] # The home team is stored
            away_team = fixture["AwayTeam"] # The away team is stored

            if home_team not in current_ratings: # If the home team does not yet have an initial rating
                current_ratings[home_team] = fixture["HomeElo"] # The first precomputed HomeElo is used as the starting rating for that team
            if away_team not in current_ratings: # If the away team does not yet have an initial rating
                current_ratings[away_team] = fixture["AwayElo"] # The first precomputed AwayElo is used as the starting rating for that team

            home_rating = current_ratings.get(home_team, pd.NA) # The current home Elo rating is looked up from running ratings
            away_rating = current_ratings.get(away_team, pd.NA) # The current away Elo rating is looked up from running ratings
            if pd.isna(home_rating): # If the home Elo rating is missing
                raise ValueError(f"Missing Elo rating value for home team: {home_team} on {fixture['Date']} vs {away_team}") # A value error is raised
            if pd.isna(away_rating): # If the away Elo rating is missing
                raise ValueError(f"Missing Elo rating value for away team: {away_team} on {fixture['Date']} vs {home_team}") # A value error is raised

            test_df.at[idx, "HomeElo"] = home_rating # The latest home Elo before the fixture is stored at the fixture index of test_df
            test_df.at[idx, "AwayElo"] = away_rating # The latest away Elo before the fixture is stored at the fixture index of test_df
            test_df.at[idx, "EloTierHome"] = 1 if home_rating >= 1900 else 2 if home_rating >= 1800 else 3 if home_rating >= 1700 else 4 # The latest Elo tier for the home side before the fixture is stored at the fixture index of test_df
            test_df.at[idx, "EloTierAway"] = 1 if away_rating >= 1900 else 2 if away_rating >= 1800 else 3 if away_rating >= 1700 else 4 # The latest Elo tier for the away side before the fixture is stored at the fixture index of test_df

            hfa_value = current_hfa # The HFA value for that fixture date is read
            if pd.isna(hfa_value): # If HFA is missing for the fixture
                raise ValueError(f"Missing HFA for 2023 fixture: {fixture['Date']} {home_team} vs {away_team}") # A value error is raised
            test_df.at[idx, "HFA"] = hfa_value # The HFA value before the fixture is stored at the fixture index of test_df

            row_features = test_df.loc[[idx], feature_columns] # The features of the fixture row being predicted are stored in a dataframe only containing the features of that single row
            raw_prediction = model.predict(row_features)[0] # The model is run on that one fixture and a prediction is generated
            predicted_result = prediction_to_result(raw_prediction) if prediction_to_result else int(raw_prediction) # The raw prediction is converted into the expected encoded result value
            predictions_by_index[idx] = predicted_result # The predicted result is stored under the row index

            if predicted_result == 1: # If prediction is home win
                score_home = 1 # The score result is 1
            elif predicted_result == 0: # If prediction is draw
                score_home = 0.5 # The score result is 0.5
            else: # If prediction is home loss
                score_home = 0 # The score result is 0

            home_rating_adjusted = home_rating + hfa_value # The elo rating for the home side is adjusted to include the home field advantage
            expected_home = 1 / (1 + 10 ** (-(home_rating_adjusted - away_rating) / 400)) # The home win probability is computed
            delta_home = k_factor * (score_home - expected_home) # The Elo change for the home side is calculated
            current_ratings[home_team] = home_rating + delta_home # The new Elo rating for the home side is calculated and is updated in the current_ratings dictionary
            current_ratings[away_team] = away_rating - delta_home # The new Elo rating for the away side is calculated and is updated in the current_ratings dictionary
            day_delta_sum += delta_home # The home Elo change is added to the daily delta sum

        championship_matches_today = championship_2023[championship_2023["DateKey"] == match_date].sort_values(["HomeTeam", "AwayTeam"]) # The championship fixtures for that date are selected and sorted by home team and then away team names
        for _, fixture in championship_matches_today.iterrows(): # Iterates over each championship fixture on that date
            home_team = fixture["HomeTeam"] # The home team is stored
            away_team = fixture["AwayTeam"] # The away team is stored
            fixture_date = fixture["DateParsed"] # The championship fixture date is stored
            if home_team not in current_ratings: # If the home team is not found in the Elo ratings map
                team_elo = elo_history[elo_history["Team"] == home_team] # The Elo history for the home team is selected
                prior_elo = team_elo[team_elo["To"] <= fixture_date] if not team_elo.empty and not pd.isna(fixture_date) else pd.DataFrame() # Only Elo records up to and including the fixture date are kept
                if prior_elo.empty: # If there are no Elo records before that fixture date
                    raise KeyError(f"Missing Elo rating entry for home team: {home_team} on {fixture['Date']} vs {away_team}") # A key error is raised
                closest_idx = prior_elo["To"].idxmax() # The index of the most recent Elo record before the match date is selected
                current_ratings[home_team] = prior_elo.loc[closest_idx, "Elo"] # The Elo rating at the closest index is stored as the current rating for the home team
            if away_team not in current_ratings: # If the away team is not found in the Elo ratings map
                team_elo = elo_history[elo_history["Team"] == away_team] # The Elo history for the away team is selected
                prior_elo = team_elo[team_elo["To"] <= fixture_date] if not team_elo.empty and not pd.isna(fixture_date) else pd.DataFrame() # Only Elo records up to and including the fixture date are kept
                if prior_elo.empty: # If there are no Elo records before that fixture date
                    raise KeyError(f"Missing Elo rating entry for away team: {away_team} on {fixture['Date']} vs {home_team}") # A key error is raised
                closest_idx = prior_elo["To"].idxmax() # The index of the most recent Elo record before the match date is selected
                current_ratings[away_team] = prior_elo.loc[closest_idx, "Elo"] # The Elo rating at the closest index is stored as the current rating for the away team
            home_rating = current_ratings[home_team] # The current Elo rating for the home team is stored
            away_rating = current_ratings[away_team] # The current Elo rating for the away team is stored
            if pd.isna(home_rating): # If the home Elo rating is missing
                raise ValueError(f"Missing Elo rating value for home team: {home_team} on {fixture['Date']} vs {away_team}") # A value error is raised
            if pd.isna(away_rating): # If the away Elo rating is missing
                raise ValueError(f"Missing Elo rating value for away team: {away_team} on {fixture['Date']} vs {home_team}") # A value error is raised

            if fixture["ResultCode"] == "H": # If the home side won
                score_home = 1 # The score result is 1
            elif fixture["ResultCode"] == "D": # If the home side drew
                score_home = 0.5 # The score result is 0.5
            else: # If the home side lost
                score_home = 0 # The score result is 0

            home_rating_adjusted = home_rating + current_hfa # The elo rating for the home side is adjusted to include the home field advantage
            expected_home = 1 / (1 + 10 ** (-(home_rating_adjusted - away_rating) / 400)) # The home win probability is computed
            delta_home = k_factor * (score_home - expected_home) # The Elo change for the home side is calculated
            current_ratings[home_team] = home_rating + delta_home # The new Elo rating for the home side is calculated and is updated in the current_ratings dictionary
            current_ratings[away_team] = away_rating - delta_home # The new Elo rating for the away side is calculated and is updated in the current_ratings dictionary
            day_delta_sum += delta_home # The home Elo change is added to the daily delta sum

        current_hfa += day_delta_sum * 0.075 # The dynamic HFA value is updated at the end of the day using the daily home delta sum

    predictions = pd.Series(predictions_by_index).reindex(test_df.index) # A series is built from the predictions_by_index dictionary and is reindexed so that row indexes match the row index of test_df
    return predictions.to_numpy(), test_df.drop(columns=["DateParsed"]) # The predictions array and test_df without the DateParsed column are returned