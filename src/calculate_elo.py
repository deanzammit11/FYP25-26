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
    margin_training_df = pd.read_csv("data/processed/eng1_all_seasons.csv") # The combined seasons dataset is read from the specified direcotory
    margin_training_df["Season"] = pd.to_numeric(margin_training_df["Season"], errors="coerce") # The Season column is converted to numeric
    margin_training_df = margin_training_df[margin_training_df["Season"] < 2023].copy() # Only rows where the season is less than 2023 are kept
    margin_data = margin_training_df[["HomeTeam", "AwayTeam", "FullTimeHomeGoals", "FullTimeAwayGoals"]].copy() # Only columns needed for goal margin are kept
    margin_data["FullTimeHomeGoals"] = pd.to_numeric(margin_data["FullTimeHomeGoals"], errors="coerce") # The Home goals column is converted to numeric
    margin_data["FullTimeAwayGoals"] = pd.to_numeric(margin_data["FullTimeAwayGoals"], errors="coerce") # The Away goals column is converted to numeric
    margin_data = margin_data.dropna(subset=["HomeTeam", "AwayTeam", "FullTimeHomeGoals", "FullTimeAwayGoals"]) # Rows with missing team names or goals are dropped
    margin_data["Margin"] = (margin_data["FullTimeHomeGoals"] - margin_data["FullTimeAwayGoals"]).abs().astype(int) # The absolute score margin is computed and stored as an integer in a new Margin column
    if margin_data.empty: # If there are no historical matches
        raise ValueError("No valid training scoreline history found to build head to head margins.") # A value error is raised
    margin_counts = margin_data.groupby(["HomeTeam", "AwayTeam", "Margin"]).size().reset_index(name="Count") # Rows are grouped by home team, away team and margin and the number of matches per group are counted and the index is then reset producing a new Count column
    margin_counts = margin_counts.sort_values(["HomeTeam", "AwayTeam", "Count", "Margin"], ascending=[True, True, False, True]) # Rows are first sorted by home team in alphabetical order then by away team in alphabetical order then by the most frequent margin and finally by the smallest margin
    most_common_margin = margin_counts.drop_duplicates(subset=["HomeTeam", "AwayTeam"], keep="first") # The most common margin row is kept for each head to head pair with the smallest margin being kept if there is a tie
    directed_margin_lookup = {(row["HomeTeam"], row["AwayTeam"]): float(row["Margin"]) for _, row in most_common_margin.iterrows()} # For each row in most_common_margin a dictionary entry is built in the format (HomeTeam,AwayTeam): Margin and stored in a dictionary
    default_margin = float(margin_data["Margin"].mode().iloc[0]) # The most frequent margin in the data is saved as a float with the smallest margin being taken if there is a tie
    non_draw_margins = margin_data[margin_data["Margin"] >= 1].copy() # Only margins which are not a draw margin are saved
    if non_draw_margins.empty: # If there are no non draw margins
        raise ValueError("No non draw training matches found to build margin probabilities.") # A value error is raised
    default_non_draw_margin = float(non_draw_margins["Margin"].mode().iloc[0]) # The most frequent non draw margin is saved as a float with the smallest margin being taken if there is a tie
    margin_probability_table = non_draw_margins["Margin"].value_counts(normalize=True) # The number of appearances of each unique margin is counted and divided by the total number of times where the margin was not 0
    margin_probability_lookup = {float(margin): float(probability) for margin, probability in margin_probability_table.items()} # For each margin a dictionary entry is built in the format margin: probability and stored in a dictionary as a float
    default_p_margin = float(margin_probability_table.min()) # Unseen margins are set to the least common margin
    normalization_numerator = sum((margin ** 0.5) * probability for margin, probability in margin_probability_lookup.items()) # For each margin and respective probability sqrt(j) * p_margin(j) is calculated and all the sum is then calculated
    if normalization_numerator <= 0: # If the numerator is invalid
        raise ValueError("Invalid numerator computed from margin probabilities.") # A value error is raised
    estimated_margins = [] # A list which will store the estimated margin for each fixture row is initialised
    p_margin_values = [] # A list which will store the p_margin value for each fixture row is initialised
    for home_team, away_team in zip(df["HomeTeam"], df["AwayTeam"]): # For each team pair in each fixture in the dataframe
        margin_value = directed_margin_lookup.get((home_team, away_team), default_margin) # The most common margin for the team pair is looked up and if it is not found the most common margin is used
        if pd.isna(margin_value): # If margin is found but is NA
            margin_value = default_margin # Most common margin is assigned
        margin_value = float(margin_value) # Margin is saved as a float
        if margin_value < 0: # If the margin is negative
            raise ValueError(f"Negative estimated margin for fixture {home_team} vs {away_team}: {margin_value}") # A value error is raised
        estimated_margins.append(margin_value) # The estimated margin is appended to the list
        p_margin_values.append(float(margin_probability_lookup.get(margin_value, default_p_margin))) # The probability of the margin is looked up and if it is found it is appended and if it is not found the least common margin probability is appended
    df["EstimatedMargin"] = estimated_margins # Estimated margin is added to the dataframe
    df["PMargin"] = p_margin_values # Margin probability is added to the dataframe

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
            if score_home == 1: # If the home side won
                p_1x2 = expected_home # Probability of result is equal to the probability of home side win
            elif score_home == 0: # If the home side lost
                p_1x2 = 1 - expected_home # Probability of result is equal to the probability of 1 - home side win
            else: # If the home side drew
                p_1x2 = 1.0 # The probability of result is set to a neutral value
            if score_home == 0.5: # If the home side drew
                s_norm = 1.0 # The normalisation term S is set to a neutral value
            else: # If the home side won or lost
                s_norm = normalization_numerator / max(float(p_1x2), 1e-12) # The normalisation term S is computed
            delta_elo_1x2 = k_factor * (score_home - expected_home) # The basic Elo change for the home side is calculated
            if score_home == 0.5: # If the home side drew
                delta_home = delta_elo_1x2 # The basic elo change is kept
            else: # If the home side won or lost
                margin_value = directed_margin_lookup.get((home_team, away_team), default_margin) # The most common margin for the team pair is looked up and if it is not found the most common margin is used
                if pd.isna(margin_value): #  If margin is found but is NA
                    margin_value = default_margin # Most common margin is assigned
                margin_value = float(margin_value) # Margin is saved as a float
                if margin_value < 0: # If the margin is negative
                    raise ValueError(f"Negative estimated margin for fixture {home_team} vs {away_team}: {margin_value}") # A value error is raised
                if margin_value < 1: # If the margin is less than 1
                    margin_value = default_non_draw_margin # The most common non draw margin is assigned
                delta_elo_goal = delta_elo_1x2 / max(float(s_norm), 1e-12) # Basic elo change is converted to a one goal equivalent with the denominator being 1e-12 if it is too small to prevent division by 0
                delta_home = delta_elo_goal * (margin_value ** 0.5) # Final elo change is calculated by multiplying by the square root of the actual margin
            elo_ratings[home_team] = home_rating + delta_home # The new Elo rating for the home side is calculated
            elo_ratings[away_team] = away_rating - delta_home # The new Elo rating for the away side is calculated
            day_delta_sum += delta_elo_1x2 # The basic home Elo change is added to the daily delta sum

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
            if score_home == 1: # If the home side won
                p_1x2 = expected_home # Probability of result is equal to the probability of home side win
            elif score_home == 0: # If the home side lost
                p_1x2 = 1 - expected_home # Probability of result is equal to the probability of 1 - home side win
            else: # If the home side drew
                p_1x2 = 1.0 # The probability of result is set to a neutral value
            if score_home == 0.5: # If the home side drew
                s_norm = 1.0 # The normalisation term S is set to a neutral value
            else: # If the home side won or lost
                s_norm = normalization_numerator / max(float(p_1x2), 1e-12) # The normalisation term S is computed
            delta_elo_1x2 = k_factor * (score_home - expected_home) # The basic Elo change for the home side is calculated
            if score_home == 0.5: # If the home side drew
                delta_home = delta_elo_1x2 # The basic elo change is kept
            else: # If the home side won or lost
                margin_value = directed_margin_lookup.get((home_team, away_team), default_margin) # The most common margin for the team pair is looked up and if it is not found the most common margin is used
                if pd.isna(margin_value): #  If margin is found but is NA
                    margin_value = default_margin # Most common margin is assigned
                margin_value = float(margin_value) # Margin is saved as a float
                if margin_value < 0: # If the margin is negative
                    raise ValueError(f"Negative estimated margin for fixture {home_team} vs {away_team}: {margin_value}") # A value error is raised
                if margin_value < 1: # If the margin is less than 1
                    margin_value = default_non_draw_margin # The most common non draw margin is assigned
                delta_elo_goal = delta_elo_1x2 / max(float(s_norm), 1e-12) # Basic elo change is converted to a one goal equivalent with the denominator being 1e-12 if it is too small to prevent division by 0
                delta_home = delta_elo_goal * (margin_value ** 0.5) # Final elo change is calculated by multiplying by the square root of the actual margin
            elo_ratings[home_team] = home_rating + delta_home # The new Elo rating for the home side is calculated
            elo_ratings[away_team] = away_rating - delta_home # The new Elo rating for the away side is calculated
            day_delta_sum += delta_elo_1x2 # The basic home Elo change is added to the daily delta sum

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
    margin_training_df = pd.read_csv("data/processed/eng1_all_seasons.csv") # The combined seasons dataset is read from the specified direcotory
    margin_training_df["Season"] = pd.to_numeric(margin_training_df["Season"], errors="coerce") # The Season column is converted to numeric
    margin_training_df = margin_training_df[margin_training_df["Season"] < 2023].copy() # Only rows where the season is less than 2023 are kept
    margin_data = margin_training_df[["HomeTeam", "AwayTeam", "FullTimeHomeGoals", "FullTimeAwayGoals"]].copy() # Only columns needed for goal margin are kept
    margin_data["FullTimeHomeGoals"] = pd.to_numeric(margin_data["FullTimeHomeGoals"], errors="coerce") # The Home goals column is converted to numeric
    margin_data["FullTimeAwayGoals"] = pd.to_numeric(margin_data["FullTimeAwayGoals"], errors="coerce") # The Away goals column is converted to numeric
    margin_data = margin_data.dropna(subset=["HomeTeam", "AwayTeam", "FullTimeHomeGoals", "FullTimeAwayGoals"]) # Rows with missing team names or goals are dropped
    margin_data["Margin"] = (margin_data["FullTimeHomeGoals"] - margin_data["FullTimeAwayGoals"]).abs().astype(int) # The absolute score margin is computed and stored as an integer in a new Margin column
    if margin_data.empty: # If there are no historical matches
        raise ValueError("No valid training scoreline history found to build head to head margins.") # A value error is raised
    margin_counts = margin_data.groupby(["HomeTeam", "AwayTeam", "Margin"]).size().reset_index(name="Count") # Rows are grouped by home team, away team and margin and the number of matches per group are counted and the index is then reset producing a new Count column
    margin_counts = margin_counts.sort_values(["HomeTeam", "AwayTeam", "Count", "Margin"], ascending=[True, True, False, True]) # Rows are first sorted by home team in alphabetical order then by away team in alphabetical order then by the most frequent margin and finally by the smallest margin
    most_common_margin = margin_counts.drop_duplicates(subset=["HomeTeam", "AwayTeam"], keep="first") # The most common margin row is kept for each head to head pair with the smallest margin being kept if there is a tie
    directed_margin_lookup = {(row["HomeTeam"], row["AwayTeam"]): float(row["Margin"]) for _, row in most_common_margin.iterrows()} # For each row in most_common_margin a dictionary entry is built in the format (HomeTeam,AwayTeam): Margin and stored in a dictionary
    default_margin = float(margin_data["Margin"].mode().iloc[0]) # The most frequent margin in the data is saved as a float with the smallest margin being taken if there is a tie
    non_draw_margins = margin_data[margin_data["Margin"] >= 1].copy() # Only margins which are not a draw margin are saved
    if non_draw_margins.empty: # If there are no non draw margins
        raise ValueError("No non draw training matches found to build margin probabilities.") # A value error is raised
    default_non_draw_margin = float(non_draw_margins["Margin"].mode().iloc[0]) # The most frequent non draw margin is saved as a float with the smallest margin being taken if there is a tie
    margin_probability_table = non_draw_margins["Margin"].value_counts(normalize=True) # The number of appearances of each unique margin is counted and divided by the total number of times where the margin was not 0
    margin_probability_lookup = {float(margin): float(probability) for margin, probability in margin_probability_table.items()} # For each margin a dictionary entry is built in the format margin: probability and stored in a dictionary as a float
    default_p_margin = float(margin_probability_table.min()) # Unseen margins are set to the least common margin
    normalization_numerator = sum((margin ** 0.5) * probability for margin, probability in margin_probability_lookup.items()) # For each margin and respective probability sqrt(j) * p_margin(j) is calculated and all the sum is then calculated
    if normalization_numerator <= 0: # If the numerator is invalid
        raise ValueError("Invalid numerator computed from margin probabilities.") # A value error is raised

    season_sorted = test_df.sort_values(["DateParsed", "HomeTeam", "AwayTeam"]) # Rows are first sorted in chronological order by date and they are then sorted by HomeTeam and AwayTeam
    season_sorted["DateKey"] = season_sorted["DateParsed"].dt.normalize() # Each time in DateParsed is set to midnight for consistency
    predictions_by_index = {} # An empty dictionary which will later store the predicted class for each row index is initialised
    probabilities_by_index = {} # An empty dictionary which will later store the predicted probabilities for each row index is initialised
    current_ratings = {} # An empty dictionary which will store the latest Elo rating for each team is initialised
    if season_sorted["HFA"].isna().all(): # If all HFA values are missing for 2023 fixtures
        raise ValueError("Missing HFA values for all 2023 fixtures in test_df") # A value error is raised
    current_hfa = season_sorted["HFA"].dropna().iloc[0] # The initial HFA value is set to the value of the first 2023 fixture

    model_classes = model.classes_ # The possible outcome labels are stored in an array
    class_to_result = {} # An empty dictionary which will later store the class labels mapped to -1,0,1
    for raw_class in model_classes: # For each unconverted class label
        mapped_result = prediction_to_result(raw_class) if prediction_to_result else int(raw_class) # If the class label needs to be converted the class label is converted else it is type casted to integer
        class_to_result[int(raw_class)] = int(mapped_result) # The mapping is stored in the dictionary

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
            margin_value = directed_margin_lookup.get((home_team, away_team), default_margin) # The most common margin for the team pair is looked up and if it is not found the most common margin is used
            if pd.isna(margin_value): # If margin is found but is NA
                margin_value = default_margin # Most common margin is assigned
            margin_value = float(margin_value) # Margin is saved as a float
            if margin_value < 0: # If the margin is negative
                raise ValueError(f"Negative estimated margin for fixture {home_team} vs {away_team}: {margin_value}") # A value error is raised
            test_df.at[idx, "EstimatedMargin"] = margin_value # Estimated margin is added to the dataframe at the fixture index of test_df
            test_df.at[idx, "PMargin"] = float(margin_probability_lookup.get(margin_value, default_p_margin)) # Margin probability is added to the dataframe at the fixture index of test_df

            row_features = test_df.loc[[idx], feature_columns] # The features of the fixture row being predicted are stored in a dataframe only containing the features of that single row
            if not hasattr(model, "predict_proba"): # If the model does not support probability estimates
                raise ValueError(f"Model {type(model).__name__} does not support predict_proba.") # A value error is raised
            probability_row = model.predict_proba(row_features)[0] # The class probabilities for that fixture are generated
            if model_classes is None: # If class labels are missing
                raise ValueError("Model supports predict_proba but fitted class labels (classes_) are missing.") # A value error is raised
            probability_map = {-1: 0.0, 0: 0.0, 1: 0.0} # A temporaty dictionary which will later the store class probabilities for the match is initialised with a probability of 0 for each class
            for raw_class, class_probability in zip(model_classes, probability_row): # For each class and class probability in the class and probability pairs
                encoded_result = class_to_result[int(raw_class)] # The model label is mapped to the target label
                probability_map[encoded_result] = float(class_probability) # The probability for that class is stored in the probability_map dictionary
            probabilities_by_index[idx] = probability_map # The class probabilities for that row index are stored
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
            if score_home == 1: # If the home side won
                p_1x2 = expected_home # Probability of result is equal to the probability of home side win
            elif score_home == 0: # If the home side lost
                p_1x2 = 1 - expected_home # Probability of result is equal to the probability of 1 - home side win
            else: # If the home side drew
                p_1x2 = 1.0 # The probability of result is set to a neutral value
            test_df.at[idx, "P1X2"] = float(p_1x2) # The proability of result is added to the dataframe at the fixture index of test_df
            if score_home == 0.5: # If the home side drew
                s_norm = 1.0 # The normalisation term S is set to a neutral value
            else: # If the home side won or lost
                s_norm = normalization_numerator / max(float(p_1x2), 1e-12) # The normalisation term S is computed
            test_df.at[idx, "SNorm"] = float(s_norm) # The normalisation term S is added to the dataframe at the fixture index of test_df
            delta_elo_1x2 = k_factor * (score_home - expected_home) # The basic Elo change for the home side is calculated
            if score_home == 0.5: # If the home side drew
                delta_elo_goal = delta_elo_1x2 # The basic elo change is kept
                delta_home = delta_elo_1x2 # The basic elo change is kept
            else: # If the home side won or lost
                if margin_value < 1: # If the margin is less than 1
                    margin_for_scaling = default_non_draw_margin # The most common non draw margin is assigned
                else: # If the margin is 1 or above
                    margin_for_scaling = float(margin_value) # Margin is saved as a float
                delta_elo_goal = delta_elo_1x2 / max(float(s_norm), 1e-12) # Basic elo change is converted to a one goal equivalent with the denominator being 1e-12 if it is too small to prevent division by 0
                delta_home = delta_elo_goal * (margin_for_scaling ** 0.5) # Final elo change is calculated by multiplying by the square root of the actual margin
            test_df.at[idx, "DeltaElo1X2"] = float(delta_elo_1x2) # The basic elo change is added to the dataframe at the fixture index of test_df
            test_df.at[idx, "DeltaEloGoal"] = float(delta_elo_goal) # The basic elo change converted to a one goal equivalent is added to the dataframe at the fixture index of test_df
            test_df.at[idx, "DeltaEloMargin"] = float(delta_home) # The final elo change is added to the dataframe at the fixture index of test_df
            current_ratings[home_team] = home_rating + delta_home # The new Elo rating for the home side is calculated and is updated in the current_ratings dictionary
            current_ratings[away_team] = away_rating - delta_home # The new Elo rating for the away side is calculated and is updated in the current_ratings dictionary
            day_delta_sum += delta_elo_1x2 # The basic home Elo change is added to the daily delta sum

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
            if score_home == 1: # If the home side won
                p_1x2 = expected_home # Probability of result is equal to the probability of home side win
            elif score_home == 0: # If the home side lost
                p_1x2 = 1 - expected_home # Probability of result is equal to the probability of 1 - home side win
            else: # If the home side drew
                p_1x2 = 1.0 # The probability of result is set to a neutral value
            if score_home == 0.5: # If the home side drew
                s_norm = 1.0 # The normalisation term S is set to a neutral value
            else: # If the home side won or lost
                s_norm = normalization_numerator / max(float(p_1x2), 1e-12) # The normalisation term S is computed
            delta_elo_1x2 = k_factor * (score_home - expected_home) # The basic Elo change for the home side is calculated
            if score_home == 0.5: # If the home side drew
                delta_home = delta_elo_1x2 # The basic elo change is kept
            else: # If the home side won or lost
                margin_value = directed_margin_lookup.get((home_team, away_team), default_margin) # The most common margin for the team pair is looked up and if it is not found the most common margin is used
                if pd.isna(margin_value): # If margin is found but is NA
                    margin_value = default_margin # Most common margin is assigned
                margin_value = float(margin_value) # Margin is saved as a float
                if margin_value < 0: # If the margin is negative
                    raise ValueError(f"Negative estimated margin for fixture {home_team} vs {away_team}: {margin_value}") # A value error is raised
                if margin_value < 1: # If the margin is less than 1
                    margin_value = default_non_draw_margin # The most common non draw margin is assigned
                delta_elo_goal = delta_elo_1x2 / max(float(s_norm), 1e-12) # Basic elo change is converted to a one goal equivalent with the denominator being 1e-12 if it is too small to prevent division by 0
                delta_home = delta_elo_goal * (margin_value ** 0.5) # Final elo change is calculated by multiplying by the square root of the actual margin
            current_ratings[home_team] = home_rating + delta_home # The new Elo rating for the home side is calculated and is updated in the current_ratings dictionary
            current_ratings[away_team] = away_rating - delta_home # The new Elo rating for the away side is calculated and is updated in the current_ratings dictionary
            day_delta_sum += delta_elo_1x2 # The basic home Elo change is added to the daily delta sum

        current_hfa += day_delta_sum * 0.075 # The dynamic HFA value is updated at the end of the day using the daily home delta sum

    predictions = pd.Series(predictions_by_index).reindex(test_df.index) # A series is built from the predictions_by_index dictionary and is reindexed so that row indexes match the row index of test_df
    if probabilities_by_index: # If probabilities were computed
        probabilities = pd.DataFrame.from_dict(probabilities_by_index, orient="index").reindex(test_df.index) # The dictionary is converted into a dataframe with the outer keys becoming the dataframe index and the inner keys becoming the dataframe columns the rows are then reindexed to match the index of the test dataframe
        test_df["ProbHome"] = probabilities[1].to_numpy() # The home win probability is stored
        test_df["ProbDraw"] = probabilities[0].to_numpy() # The draw probability is stored
        test_df["ProbAway"] = probabilities[-1].to_numpy() # The away win probability is stored
    return predictions.to_numpy(), test_df.drop(columns=["DateParsed"]) # The predictions array and test_df without the DateParsed column are returned