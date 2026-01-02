import pandas as pd
from src.utils import ensure_dirs, save_csv
from pathlib import Path

def prepare_features():
    ensure_dirs("data/features") # Checks if directory exists and if it does not it creates it
    df = pd.read_csv("data/processed/eng1_all_seasons.csv") # Csv file is read from the specified directory and stored in a data frame

    df = df.dropna(subset=["FullTimeResult"]) # Drop rows with missing results

    df["ResultEncoded"] = df["FullTimeResult"].map({"H": 1, "D": 0, "A": -1}) # Encode target
    df["HomePoints"] = df["ResultEncoded"].map({1: 3, 0: 1, -1: 0}) # Result is mapped to points
    df["AwayPoints"] = df["ResultEncoded"].map({1: 0, 0: 1, -1: 3}) # Result is mapped to points
    
    # Betting Odds Features
    # Bet365 odds are converted into implied probabilities
    df["Home"] = 1 / df["Bet365HomeWinOdds"]
    df["Draw"] = 1 / df["Bet365DrawOdds"]
    df["Away"] = 1 / df["Bet365AwayWinOdds"]

    df["Total"] = df["Home"] + df["Draw"] + df["Away"] # Calculate the total of the implied probabilities

    df["Bet365HomeWinOddsPercentage"] = df["Home"] / df["Total"] # Convert the implied home win probability into a percentage
    df["Bet365DrawOddsPercentage"] = df["Draw"] / df["Total"] # Convert the implied draw probability into a percentage
    df["Bet365AwayWinOddsPercentage"] = df["Away"] / df["Total"] # Convert the implied away win probability into a percentage

    df["OddsDifference_HvA"] = (df["Bet365HomeWinOddsPercentage"] - df["Bet365AwayWinOddsPercentage"]) # Home vs away odds difference without absolute value
    df["OddsDifference_HvD"] = (df["Bet365HomeWinOddsPercentage"] - df["Bet365DrawOddsPercentage"]) # Home vs draw odds difference without absolute value
    df["OddsDifference_AvD"] = (df["Bet365AwayWinOddsPercentage"] - df["Bet365DrawOddsPercentage"]) # Away vs draw odds difference without absolute value
    # df["OddsDifference_HvA"] = (df["Bet365HomeWinOddsPercentage"] - df["Bet365AwayWinOddsPercentage"]).abs() # Home vs away odds difference with absolute value
    # df["OddsDifference_HvD"] = (df["Bet365HomeWinOddsPercentage"] - df["Bet365DrawOddsPercentage"]).abs() # Home vs draw odds difference with absolute value
    # df["OddsDifference_AvD"] = (df["Bet365AwayWinOddsPercentage"] - df["Bet365DrawOddsPercentage"]).abs() # Away vs draw odds difference with absolute value

    odds_cols = [ # Columns to store in odds csv file are selected
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "Bet365HomeWinOddsPercentage", "Bet365DrawOddsPercentage", "Bet365AwayWinOddsPercentage",
        "OddsDifference_HvA", "OddsDifference_HvD", "OddsDifference_AvD"
    ]
    df_odds = df[odds_cols].dropna() # Null values are removed
    save_csv(df_odds, "data/features/eng1_data_odds.csv") # Csv without null values is saved into the specified directory

    # Form Features
    df["HomeForm"] = (
        df["HomePoints"].groupby(df["HomeTeam"]).shift() # Each Home team's home results is shifted one game back
        .groupby(df["HomeTeam"]).rolling(window=5, min_periods=1).sum() # Sum of points scored by the home team over the last 5 home games is computed
        .reset_index(level=0, drop=True) # The HomeTeam grouping in the multi level index is dropped leaving only the row id
    )
    df["AwayForm"] = (
        df["AwayPoints"].groupby(df["AwayTeam"]).shift() # Each Away team's away results is shifted one game back
        .groupby(df["AwayTeam"]).rolling(window=5, min_periods=1).sum() # Sum of points scored by the away team over the last 5 away games is computed
        .reset_index(level=0, drop=True) # The AwayTeam grouping in the multi level index is dropped leaving only the row id
    )

    df["HomeAdvantageIndex"] = df["HomeForm"] - df["AwayForm"] # Calculates home advantage based on how the home side performs at home and how the away side performs away from home

    home_results = df[["Season", "Date", "HomeTeam", "HomePoints"]].rename(columns={"HomeTeam": "Team", "HomePoints": "Points"}) # A table storing the the points scored by the home team per fixture is created
    home_results["MatchIndex"] = df.index # Adds the row number to the home results
    away_results = df[["Season", "Date", "AwayTeam", "AwayPoints"]].rename(columns={"AwayTeam": "Team", "AwayPoints": "Points"}) # A table storing the the points scored by the away team per fixture is created
    away_results["MatchIndex"] = df.index # Adds the row number to the away results

    all_results = pd.concat([home_results, away_results], ignore_index=True) # Concatenates the home and away results into 1 results dataframe
    all_results = all_results.sort_values(["Season", "Team", "MatchIndex"]).drop(columns="MatchIndex") # Both tables are joined into one table and sorted by season, team name, and match order and the MatchIndex is then dropped

    grouped_points = all_results.groupby(["Season", "Team"])["Points"].shift() # Points scored are grouped by season and team and shifted one game back
    
    all_results["GeneralForm"] = (
        grouped_points.groupby([all_results["Season"], all_results["Team"]]) # Points are grouped again by Season and Team
        .rolling(window=5, min_periods=1).sum() # Sum of points scored over the last 5 games is computed
        .reset_index(level=[0, 1], drop=True) # The Season and Team grouping in the multi level index is dropped leaving only the row id
    )

    df = df.merge(
        all_results[["Season", "Team", "Date", "GeneralForm"]], # Merge df with the Season, Team, Date and GeneralForm columns in all_results
        how="left", # Keeps all columns in df and only merges the columns from all_results
        left_on=["Season", "HomeTeam", "Date"], right_on=["Season", "Team", "Date"] # Match Season, HomeTeam and Date from the left table and Season, Team and Date from the right table and if they match merge
    ).rename(columns={"GeneralForm": "HomeGeneralForm"}).drop(columns="Team") # GeneralForm column is renamed and the new Team column is removed

    df = df.merge(
        all_results[["Season", "Team", "Date", "GeneralForm"]], # Merge df with the Season, Team, Date and GeneralForm columns in all_results
        how="left", # Keeps all columns in df and only merges the columns from all_results
        left_on=["Season", "AwayTeam", "Date"], right_on=["Season", "Team", "Date"] # Match Season, AwayTeam and Date from the left table and Season, Team and Date from the right table and if they match merge
    ).rename(columns={"GeneralForm": "AwayGeneralForm"}).drop(columns="Team") # GeneralForm column is renamed and the new Team column is removed

    df["GeneralFormDifference"] = df["HomeGeneralForm"] - df["AwayGeneralForm"] # Points scored by the away team in the last 5 matches are subtracted from the points scored by the home team in the last 5 matches

    home_team_groups = df.groupby(["Season", "HomeTeam"]) # Rows are grouped by season and home team
    previous_home_scored = home_team_groups["FullTimeHomeGoals"].shift() # The values of the goals scored at home are shifted one row downwards
    df["AverageGoalsScoredAtHome"] = (
        previous_home_scored.groupby([df["Season"], df["HomeTeam"]]).cumsum() / home_team_groups.cumcount().replace(0, pd.NA) # The cumulative sum of the goals scored by the home team up to and excluding the current match is divded by the number of rows which have appeared in the group with zeros being replaced with NA since you cannot divide by 0
    ).fillna(0) # Null values are replaced with 0 where the first match is always 0

    previous_home_conceded = home_team_groups["FullTimeAwayGoals"].shift() # The values of the goals conceded at home are shifted one row downwards
    df["AverageGoalsConcededAtHome"] = (
        previous_home_conceded.groupby([df["Season"], df["HomeTeam"]]).cumsum() / home_team_groups.cumcount().replace(0, pd.NA) # The cumulative sum of the goals conceded by the home team up to and excluding the current match is divded by the number of rows which have appeared in the group with zeros being replaced with NA since you cannot divide by 0
    ).fillna(0) # Null values are replaced with 0 where the first match is always 0

    away_team_groups = df.groupby(["Season", "AwayTeam"]) # Rows are grouped by season and away team
    previous_away_scored = away_team_groups["FullTimeAwayGoals"].shift() # The values of the goals scored away from home are shifted one row downwards
    df["AverageGoalsScoredAtAway"] = (
        previous_away_scored.groupby([df["Season"], df["AwayTeam"]]).cumsum() / away_team_groups.cumcount().replace(0, pd.NA) # The cumulative sum of the goals scored by the away team away from home up to and excluding the current match is divded by the number of rows which have appeared in the group with zeros being replaced with NA since you cannot divide by 0
    ).fillna(0) # Null values are replaced with 0 where the first match is always 0

    previous_away_conceded = away_team_groups["FullTimeHomeGoals"].shift() # The values of the goals conceded away from home are shifted one row downwards
    df["AverageGoalsConcededAtAway"] = (
        previous_away_conceded.groupby([df["Season"], df["AwayTeam"]]).cumsum() / away_team_groups.cumcount().replace(0, pd.NA) # The cumulative sum of the goals conceded by the away team away from home up to and excluding the current match is divded by the number of rows which have appeared in the group with zeros being replaced with NA since you cannot divide by 0
    ).fillna(0) # Null values are replaced with 0 where the first match is always 0

    team_home_records = pd.DataFrame({ # A dataframe storing the season, date, team, goals scored and conceded and match index of the home team for each fixture is created
        "Season": df["Season"],
        "Date": df["Date"],
        "Team": df["HomeTeam"],
        "GoalsFor": df["FullTimeHomeGoals"],
        "GoalsAgainst": df["FullTimeAwayGoals"],
        "MatchIndex": df.index,
        "IsHome": True,
    })
    team_away_records = pd.DataFrame({ # A dataframe storing the season, date, team, goals scored and conceded and match index of the away team for each fixture is created
        "Season": df["Season"],
        "Date": df["Date"],
        "Team": df["AwayTeam"],
        "GoalsFor": df["FullTimeAwayGoals"],
        "GoalsAgainst": df["FullTimeHomeGoals"],
        "MatchIndex": df.index,
        "IsHome": False,
    })
    team_records = pd.concat([team_home_records, team_away_records], ignore_index=True) # Concatenates the home and away records into 1 records dataframe
    team_records = team_records.sort_values(["Season", "Team", "MatchIndex"]) # All records are sorted by Season, Team and MatchIndex
    team_groups = team_records.groupby(["Season", "Team"], sort=False) # All records are grouped by season and team
    team_records["TotalGoalsScored"] = team_groups["GoalsFor"].cumsum() - team_records["GoalsFor"] # The TotalGoalsScored excluding the current match are computed
    team_records["TotalGoalsConceded"] = team_groups["GoalsAgainst"].cumsum() - team_records["GoalsAgainst"] # The TotalGoalsConceded excluding the current match are computed
    team_records["Outcome"] = "Draw" # Adds an Outcome column to the Dataframe and initialises each value to Draw
    team_records.loc[team_records["GoalsFor"] > team_records["GoalsAgainst"], "Outcome"] = "Win" # If the GoalsFor are greater than GoalsAgainst for a row the Outcome is set to Win
    team_records.loc[team_records["GoalsFor"] < team_records["GoalsAgainst"], "Outcome"] = "Loss" # If the GoalsFor are smaller than GoalsAgainst for a row the Outcome is set to Loss

    win_mask = team_records["Outcome"].eq("Win") # When Outcome is Win win_mask is True else it is False
    win_block = (~win_mask).groupby([team_records["Season"], team_records["Team"]]).cumsum() # For each Season and Team group a counter which is incremented everytime a non-win occurs is created
    team_records["WinStreak"] = win_mask.groupby([team_records["Season"], team_records["Team"], win_block]).cumcount() + 1 # For each Season, Team and win_block group WinStreak is set to the cumulative count up to that point + 1
    team_records["WinStreak"] = team_records["WinStreak"].where(win_mask, 0) # Where win_mask is False WinStreak is set to 0 cancelling out non-wins

    loss_mask = team_records["Outcome"].eq("Loss") # When Outcome is Loss loss_mask is True else it is False
    loss_block = (~loss_mask).groupby([team_records["Season"], team_records["Team"]]).cumsum() # For each Season and Team group a counter which is incremented everytime a non-loss occurs is created
    team_records["LossStreak"] = loss_mask.groupby([team_records["Season"], team_records["Team"], loss_block]).cumcount() + 1 # For each Season, Team and loss_block group LossStreak is set to the cumulative count up to that point + 1
    team_records["LossStreak"] = team_records["LossStreak"].where(loss_mask, 0) # Where loss_mask is False LossStreak is set to 0 cancelling out non-losses

    team_records["WinStreakPrior"] = team_groups["WinStreak"].shift().fillna(0) # Each team's WinStreak value is shifted one game back with null values being assigned as zero
    team_records["LossStreakPrior"] = team_groups["LossStreak"].shift().fillna(0) # Each team's LossStreak value is shifted one game back with null values being assigned as zero
    
    team_records["TotalWins"] = team_records["Outcome"].eq("Win").astype(int).groupby([team_records["Season"], team_records["Team"]]).cumsum() # If the outcome in team_records is a Win 1 is returned 0 otherwise and the cumulative sum is calculated for each Season and Team group
    team_records["TotalDraws"] = team_records["Outcome"].eq("Draw").astype(int).groupby([team_records["Season"], team_records["Team"]]).cumsum() # If the outcome in team_records is a Draw 1 is returned 0 otherwise and the cumulative sum is calculated for each Season and Team group
    team_records["TotalLosses"] = team_records["Outcome"].eq("Loss").astype(int).groupby([team_records["Season"], team_records["Team"]]).cumsum() # If the outcome in team_records is a Loss 1 is returned 0 otherwise and the cumulative sum is calculated for each Season and Team group
    
    team_records["TotalWins"] = team_groups["TotalWins"].shift().fillna(0) # Each team's TotalWins value is shifted one game back with null values being assigned as zero
    team_records["TotalDraws"] = team_groups["TotalDraws"].shift().fillna(0) # Each team's TotalDraws value is shifted one game back with null values being assigned as zero
    team_records["TotalLosses"] = team_groups["TotalLosses"].shift().fillna(0) # Each team's TotalLosses value is shifted one game back with null values being assigned as zero
    
    home_totals = team_records[team_records["IsHome"]].set_index("MatchIndex") # The rows were IsHome is true are stored and their index is set to the original row number
    away_totals = team_records[~team_records["IsHome"]].set_index("MatchIndex") # The rows were IsHome is false are stored and their index is set to the original row number

    df["TotalGoalsScoredHome"] = home_totals["TotalGoalsScored"].reindex(df.index).fillna(0) # The TotalGoalsScored are stored in df while reindexing to contain the full index of df with instances where there were no prior values for that home fixture being filled with 0
    df["TotalGoalsConcededHome"] = home_totals["TotalGoalsConceded"].reindex(df.index).fillna(0) # The TotalGoalsConceded are stored in df while reindexing to contain the full index of df with instances where there were no prior values for that home fixture being filled with 0
    df["TotalGoalsScoredAway"] = away_totals["TotalGoalsScored"].reindex(df.index).fillna(0) # The TotalGoalsScored are stored in df while reindexing to contain the full index of df with instances where there were no prior values for that away fixture being filled with 0
    df["TotalGoalsConcededAway"] = away_totals["TotalGoalsConceded"].reindex(df.index).fillna(0) # The TotalGoalsConceded are stored in df while reindexing to contain the full index of df with instances where there were no prior values for that away fixture being filled with 0
    df["WinStreakHome"] = home_totals["WinStreakPrior"].reindex(df.index).fillna(0) # The WinStreakPrior for the home side is stored in df while reindexing to contain the full index of df with instances where there were no prior values for that home fixture being filled with 0
    df["LossStreakHome"] = home_totals["LossStreakPrior"].reindex(df.index).fillna(0) # The LossStreakPrior for the home side is stored in df while reindexing to contain the full index of df with instances where there were no prior values for that home fixture being filled with 0
    df["WinStreakAway"] = away_totals["WinStreakPrior"].reindex(df.index).fillna(0) # The WinStreakPrior for the away side is stored in df while reindexing to contain the full index of df with instances where there were no prior values for that home fixture being filled with 0
    df["LossStreakAway"] = away_totals["LossStreakPrior"].reindex(df.index).fillna(0) # The LossStreakPrior for the away side is stored in df while reindexing to contain the full index of df with instances where there were no prior values for that home fixture being filled with 0
    
    df["TotalWinsHome"] = home_totals["TotalWins"].reindex(df.index).fillna(0) # The TotalWins for the home side is stored in df while reindexing to contain the full index of df with instances where there were no prior values for that home fixture being filled with 0
    df["TotalDrawsHome"] = home_totals["TotalDraws"].reindex(df.index).fillna(0) # The TotalDraws for the home side is stored in df while reindexing to contain the full index of df with instances where there were no prior values for that home fixture being filled with 0
    df["TotalLossesHome"] = home_totals["TotalLosses"].reindex(df.index).fillna(0) # The TotalLosses for the home side is stored in df while reindexing to contain the full index of df with instances where there were no prior values for that home fixture being filled with 0
    df["TotalWinsAway"] = away_totals["TotalWins"].reindex(df.index).fillna(0) # The TotalWins for the away side is stored in df while reindexing to contain the full index of df with instances where there were no prior values for that away fixture being filled with 0
    df["TotalDrawsAway"] = away_totals["TotalDraws"].reindex(df.index).fillna(0) # The TotalDraws for the away side is stored in df while reindexing to contain the full index of df with instances where there were no prior values for that away fixture being filled with 0
    df["TotalLossesAway"] = away_totals["TotalLosses"].reindex(df.index).fillna(0) # The TotalLosses for the away side is stored in df while reindexing to contain the full index of df with instances where there were no prior values for that away fixture being filled with 0

    home_records = pd.DataFrame({ # A data frame consisting of 5 columns to store the records for the home side is built
        "Team": df["HomeTeam"], # Team name is the name of the home team
        "Opponent": df["AwayTeam"], # Opponent is the name of the away team
        "Points": df["HomePoints"], # Points is the number of points scored by the home team in that fixture
        "MatchIndex": df.index, # The respective row number in the original dataset
        "IsHome": True, # A boolean flag showing that the team played at home
    })
    away_records = pd.DataFrame({
        "Team": df["AwayTeam"], # Team name is the name of the away team
        "Opponent": df["HomeTeam"], # Opponent is the name of the home team
        "Points": df["AwayPoints"], # Points is the number of points scored by the away team in that fixture
        "MatchIndex": df.index, # The respective row number in the original dataset
        "IsHome": False, # A boolean flag showing that the team played away from home
    })

    head_to_head = pd.concat([home_records, away_records], ignore_index=True) # The home and away records are concatenated into a single dataframe with the index being reset.
    grouped_points = head_to_head.groupby(["Team", "Opponent"], sort=False)["Points"] # Each team, opponent pairing is grouped based on points while keeping the same order
    head_to_head["HistoricalPoints"] = (grouped_points.cumsum() - head_to_head["Points"]).fillna(0) # Computes the total points scored by a team against an opponent before the current match

    home_history = head_to_head[head_to_head["IsHome"]].set_index("MatchIndex")["HistoricalPoints"] # Selects the HistoricalPoints column for home records only and the index is replaced with the original index in df
    away_history = head_to_head[~head_to_head["IsHome"]].set_index("MatchIndex")["HistoricalPoints"] # Selects the HistoricalPoints column for away records only and the index is replaced with the original index in df

    df["HistoricalEncountersHome"] = home_history.reindex(df.index).fillna(0) # Reindexes home_history to contain the full index of df with instances where there were no previous encounters for that home fixture being filled with 0
    df["HistoricalEncountersAway"] = away_history.reindex(df.index).fillna(0) # Reindexes away_history to contain the full index of df with instances where there were no previous encounters for that away fixture being filled with 0

    form_cols = [ # Columns to store in form csv file are selected
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "HomeForm", "AwayForm", "HomeAdvantageIndex",
        "HomeGeneralForm", "AwayGeneralForm", "GeneralFormDifference",
        "AverageGoalsScoredAtHome", "AverageGoalsScoredAtAway",
        "AverageGoalsConcededAtHome", "AverageGoalsConcededAtAway",
        "TotalGoalsScoredHome", "TotalGoalsScoredAway",
        "TotalGoalsConcededHome", "TotalGoalsConcededAway",
        "WinStreakHome", "WinStreakAway",
        "LossStreakHome", "LossStreakAway",
        "TotalWinsHome", "TotalWinsAway",
        "TotalDrawsHome", "TotalDrawsAway",
        "TotalLossesHome", "TotalLossesAway",
        "HistoricalEncountersHome", "HistoricalEncountersAway"
    ]

    df_form = df[form_cols].fillna(0) # Sets missing values to 0 since form cannot be calculated on first appearance
    save_csv(df_form, "data/features/eng1_data_form.csv") # Csv without null values is saved into the specified directory

    # Ratings Features
    season_to_fifa_version = {2019: 20, 2020: 21, 2021: 22, 2022: 23, 2023: 24} # Maps the season to the respective fifa version
    season_team_name_map = { # Maps the team names from different datasets to each other by season
        2019: {
            "Liverpool": "Liverpool",
            "Man City": "Manchester City",
            "Man United": "Manchester United",
            "Chelsea": "Chelsea",
            "Leicester": "Leicester City",
            "Tottenham": "Tottenham Hotspur",
            "Wolves": "Wolverhampton Wanderers",
            "Arsenal": "Arsenal",
            "Sheffield United": "Sheffield United",
            "Burnley": "Burnley",
            "Southampton": "Southampton",
            "Everton": "Everton",
            "Newcastle": "Newcastle United",
            "Crystal Palace": "Crystal Palace",
            "Brighton": "Brighton & Hove Albion",
            "West Ham": "West Ham United",
            "Aston Villa": "Aston Villa",
            "Bournemouth": "AFC Bournemouth",
            "Watford": "Watford",
            "Norwich": "Norwich City",
        },
        2020: {
            "Liverpool": "Liverpool",
            "Man City": "Manchester City",
            "Man United": "Manchester United",
            "Chelsea": "Chelsea",
            "Leicester": "Leicester City",
            "Tottenham": "Tottenham Hotspur",
            "Wolves": "Wolverhampton Wanderers",
            "Arsenal": "Arsenal",
            "Sheffield United": "Sheffield United",
            "Burnley": "Burnley",
            "Southampton": "Southampton",
            "Everton": "Everton",
            "Newcastle": "Newcastle United",
            "Crystal Palace": "Crystal Palace",
            "Brighton": "Brighton & Hove Albion",
            "West Ham": "West Ham United",
            "Aston Villa": "Aston Villa",
            "Fulham": "Fulham",
            "Leeds": "Leeds United",
            "West Brom": "West Bromwich Albion",
        },
        2021: {
            "Liverpool": "Liverpool",
            "Man City": "Manchester City",
            "Man United": "Manchester United",
            "Chelsea": "Chelsea",
            "Leicester": "Leicester City",
            "Tottenham": "Tottenham Hotspur",
            "Wolves": "Wolverhampton Wanderers",
            "Arsenal": "Arsenal",
            "Watford": "Watford",
            "Burnley": "Burnley",
            "Southampton": "Southampton",
            "Everton": "Everton",
            "Newcastle": "Newcastle United",
            "Crystal Palace": "Crystal Palace",
            "Brighton": "Brighton & Hove Albion",
            "West Ham": "West Ham United",
            "Aston Villa": "Aston Villa",
            "Brentford": "Brentford",
            "Leeds": "Leeds United",
            "Norwich": "Norwich City",
        },
        2022: {
            "Liverpool": "Liverpool",
            "Man City": "Manchester City",
            "Man United": "Manchester United",
            "Chelsea": "Chelsea",
            "Leicester": "Leicester City",
            "Tottenham": "Tottenham Hotspur",
            "Wolves": "Wolverhampton Wanderers",
            "Arsenal": "Arsenal",
            "Nott'm Forest": "Nottingham Forest",
            "Fulham": "Fulham",
            "Southampton": "Southampton",
            "Everton": "Everton",
            "Newcastle": "Newcastle United",
            "Crystal Palace": "Crystal Palace",
            "Brighton": "Brighton & Hove Albion",
            "West Ham": "West Ham United",
            "Aston Villa": "Aston Villa",
            "Brentford": "Brentford",
            "Leeds": "Leeds United",
            "Bournemouth": "AFC Bournemouth",
        },
        2023: {
            "Liverpool": "Liverpool",
            "Man City": "Manchester City",
            "Man United": "Manchester United",
            "Chelsea": "Chelsea",
            "Luton": "Luton Town",
            "Tottenham": "Tottenham Hotspur",
            "Wolves": "Wolverhampton Wanderers",
            "Arsenal": "Arsenal",
            "Nott'm Forest": "Nottingham Forest",
            "Fulham": "Fulham",
            "Burnley": "Burnley",
            "Everton": "Everton",
            "Newcastle": "Newcastle United",
            "Crystal Palace": "Crystal Palace",
            "Brighton": "Brighton & Hove Albion",
            "West Ham": "West Ham United",
            "Aston Villa": "Aston Villa",
            "Brentford": "Brentford",
            "Sheffield United": "Sheffield United",
            "Bournemouth": "AFC Bournemouth",
        },
    }

    df["Season"] = pd.to_numeric(df["Season"], errors="coerce") # The values in the Season column are converted to numeric values with invalid values being null
    df["FifaVersion"] = df["Season"].map(season_to_fifa_version) # The Season for each row in df is mapped to the respective fifa version
    df["HFA"] = df["Season"].map({2019: 56.2, 2020: 33.4, 2021: 32.5, 2022: 45.4, 2023: 48.8}) # The HFA value is mapped to the respective season

    df["HomeTeamFifa"] = [season_team_name_map.get(season, {}).get(team, team) for team, season in zip(df["HomeTeam"], df["Season"])] # Each home team is paired with the respective season and the respective season is then found from the outer dictionary and from the inner dictionary the fifa team name is then returned for each pair in the form of a list
    df["AwayTeamFifa"] = [season_team_name_map.get(season, {}).get(team, team) for team, season in zip(df["AwayTeam"], df["Season"])] # Each away team is paired with the respective season and the respective season is then found from the outer dictionary and from the inner dictionary the fifa team name is then returned for each pair in the form of a list

    fifa = pd.read_csv("data/processed/fifa_20-24_teams_data.csv") # Fifa ratings csv file is loaded into a Dataframe
    fifa["fifa_version"] = pd.to_numeric(fifa["fifa_version"], errors="coerce") # The values in the fifa_version column are converted to numeric values with invalid values being null
    fifa = fifa[fifa["fifa_version"].isin(season_to_fifa_version.values())] # Dataframe is filtered using the previosly defined mapping
    fifa["Season"] = fifa["fifa_version"].map({v: k for k, v in season_to_fifa_version.items()}) # Reverses the mapping in season_to_fifa_version dictionary and a new mapping stored in the newly added Season column is created from Fifa version to Season
    fifa_ratings = fifa[["Season", "team_name", "overall", "attack", "midfield", "defence"]].copy() # The required columns are selected

    home_fifa_ratings = fifa_ratings.rename(columns={ # Creates a dataframe for the home ratings and renames the columns appropriately
        "team_name": "HomeTeamFifa",
        "overall": "HomeFifaOverall",
        "attack": "HomeFifaAttack",
        "midfield": "HomeFifaMidfield",
        "defence": "HomeFifaDefence",
    })
    away_fifa_ratings = fifa_ratings.rename(columns={ # Creates a dataframe for the away ratings and renames the columns appropriately
        "team_name": "AwayTeamFifa",
        "overall": "AwayFifaOverall",
        "attack": "AwayFifaAttack",
        "midfield": "AwayFifaMidfield",
        "defence": "AwayFifaDefence",
    })

    df = df.merge(home_fifa_ratings, how="left", on=["Season", "HomeTeamFifa"]) # Performs a left join from df to home_fifa_ratings matching Season and HomeTeamFifa
    df = df.merge(away_fifa_ratings, how="left", on=["Season", "AwayTeamFifa"]) # Performs a left join from df to away_fifa_ratings matching Season and AwayTeamFifa

    elo_dir = Path("data/raw/Elo Ratings") # Stores the path which points to the directory where the Elo History of each club is stored
    elo_frames = [] # An empty list which will later store the elo history in a dataframe for each club is initialised
    club_name_map = { # The club names which do not match are mapped to each other
        "Forest": "Nott'm Forest",
    }
    for path in elo_dir.glob("*.csv"): # For each csv file in the elo directory
        elo_team = pd.read_csv(path) # The csv file is read
        elo_team["From"] = pd.to_datetime(elo_team["From"], errors="coerce") # The From column is converted into datetime
        elo_team["To"] = pd.to_datetime(elo_team["To"], errors="coerce") # The To column is converted into datetime
        elo_team["Level"] = pd.to_numeric(elo_team["Level"], errors="coerce") # The Level column is converted into numeric
        elo_team["Team"] = elo_team["Club"].replace(club_name_map) # Ensures that the team names match and if they do not match they are replaced using the club name mapping
        elo_frames.append(elo_team[["Team", "Level", "Elo", "From", "To"]]) # The columns to keep are selected and are the elo history for that club is added to elo_frames

    elo = pd.concat(elo_frames, ignore_index=True) # Each seperate club dataframe is concatenated to form a single dataframe and the index is reset
    elo = elo.dropna(subset=["Team", "Elo", "From", "To"]).copy() # The rows where any of "Team", "Elo", "From", "To" columns is null are dropped and the new dataframe is copied
    elo = elo.sort_values(["Team", "From"]) # Rows are first sorted by team name and they are then sorted by From Date

    df["DateParsed"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce") # The match Date column is converted into datetime

    initial_elo_map = {} # An empty dictionary which will later store the mapping from club to the initial elo rating is initialised
    season_2023 = df[df["Season"] == 2023] # The dataframe is filterted only storing row where the Season is 2023
    if not season_2023.empty: # Checks if there are matches in the 2023 season if there are no matches the if statement is skipped
        season_teams = pd.concat([
            season_2023[["HomeTeam", "DateParsed"]].rename(columns={"HomeTeam": "Team"}), # HomeTeam and DateParsed columns are stored in a new dataframe with the HomeTeam column being renamed to Team
            season_2023[["AwayTeam", "DateParsed"]].rename(columns={"AwayTeam": "Team"}), # AwayTeam and DateParsed columns are stored in a new dataframe with the AwayTeam column being renamed to Team
        ], ignore_index=True) # The above 2 dataframes are concatenated with and resets the index with the resulting dataframe containing the appareance of each team in the 2023 season along with the match date
        first_match_dates = season_teams.groupby("Team")["DateParsed"].min() # The dataframe is grouped by each team and the first match date in 2023 for each team is selected
        for team, first_date in first_match_dates.items(): # For each team and respective date in the resulting series
            team_elo = elo[elo["Team"] == team] # The elo dataframe is filtered and only the rows for the respective club are stored
            if team_elo.empty or pd.isna(first_date): # If there is no Elo history or the first date is missing
                continue # Jump to the next step in the for loop
            prior_elo = team_elo[team_elo["To"] <= first_date] # Only the rows where the To is less than or equal to the date of the first match of the season are kept
            if prior_elo.empty: # If there are no records before the date of the first match of the season
                continue # Jump to the next step in the for loop
            closest_idx = prior_elo["To"].idxmax() # The index of the row where the value of the To column is the largest/most recent date is returned
            initial_elo_map[team] = prior_elo.loc[closest_idx, "Elo"] # The Elo rating for the row at the closest index stored in the dictionary under that team name

    home_side = df[["HomeTeam", "Season", "DateParsed"]].rename(columns={"HomeTeam": "Team"}) # A temporary table to store the elo rating for the home side consisting of HomeTeam, Season and DateParsed columns with the HomeTeam column being renamed to Team is created
    home_side["Elo"] = pd.NA # The Elo column is added and assigned nulls as a temporary value
    home_2019_22 = home_side["Season"].between(2019, 2022) # The rows where the season is between 2019 and 2022 inclusive are stored
    if home_2019_22.any(): # If there are matches which fall between the 2019-2022 seasons
        home_side_2019_22 = home_side[home_2019_22 & home_side["DateParsed"].notna()] # The dataframe is filtered for rows which fall between the 2019-2022 seasons and the DateParsed is not null
        for team, group in home_side_2019_22.groupby("Team"): # For each team and group containing all matches for that team
            team_elo = elo[elo["Team"] == team] # The elo dataframe is filtered and only the rows for the respective club are stored
            if team_elo.empty: # If there is no Elo history
                continue # Jump to the next step in the for loop
            group_sorted = group.drop(columns="Elo").sort_values("DateParsed") # The Elo column is dropped and rows are sorted by match date
            team_elo_sorted = team_elo.sort_values("To") # Elo records are sorted by their end date
            merged = pd.merge_asof(group_sorted, team_elo_sorted, left_on="DateParsed", right_on="To", direction="backward") # Compares DateParsed from group_sorted to to from team_elo_sorted and the largest/closest To value which is ≤ to the match date is picked and those 2 rows are merged
            home_side.loc[group_sorted.index, "Elo"] = merged["Elo"].values # The Elo ratings of merged are assigned to home_side based on the matching index since group_sorted is a subset of home_side
    home_2023 = home_side["Season"] == 2023 # The dataframe is filterted only storing row where the Season is 2023
    if home_2023.any(): # If there are matches which fall under the 2023 season
        home_side.loc[home_2023, "Elo"] = home_side.loc[home_2023, "Team"].map(initial_elo_map) # For the home side for every fixture falling under the 2023 season the initial Elo rating from initial_elo_map is looked up and that value is written into the Elo column setting the Elo rating for each fixture in 2023 to the initial elo rating

    away_side = df[["AwayTeam", "Season", "DateParsed"]].rename(columns={"AwayTeam": "Team"}) # A temporary table to store the elo rating for the away side consisting of AwayTeam, Season and DateParsed columns with the AwayTeam column being renamed to Team is created
    away_side["Elo"] = pd.NA # The Elo column is added and assigned nulls as a temporary value
    away_2019_22 = away_side["Season"].between(2019, 2022) # The rows where the season is between 2019 and 2022 inclusive are stored
    if away_2019_22.any(): # If there are matches which fall between the 2019-2022 seasons
        away_side_2019_22 = away_side[away_2019_22 & away_side["DateParsed"].notna()] # The dataframe is filtered for rows which fall between the 2019-2022 seasons and the DateParsed is not null
        for team, group in away_side_2019_22.groupby("Team"): # For each team and group containing all matches for that team
            team_elo = elo[elo["Team"] == team] # The elo dataframe is filtered and only the rows for the respective club are stored
            if team_elo.empty: # If there is no Elo history
                continue # Jump to the next step in the for loop
            group_sorted = group.drop(columns="Elo").sort_values("DateParsed") # The Elo column is dropped and rows are sorted by match date
            team_elo_sorted = team_elo.sort_values("To") # Elo records are sorted by their end date
            merged = pd.merge_asof(group_sorted, team_elo_sorted, left_on="DateParsed", right_on="To", direction="backward") # Compares DateParsed from group_sorted to to from team_elo_sorted and the largest/closest To value which is ≤ to the match date is picked and those 2 rows are merged
            away_side.loc[group_sorted.index, "Elo"] = merged["Elo"].values # The Elo ratings of merged are assigned to away_side based on the matching index since group_sorted is a subset of away_side
    away_2023 = away_side["Season"] == 2023 # The dataframe is filterted only storing row where the Season is 2023
    if away_2023.any(): # If there are matches which fall under the 2023 season
        away_side.loc[away_2023, "Elo"] = away_side.loc[away_2023, "Team"].map(initial_elo_map) # For the away side for every fixture falling under the 2023 season the initial Elo rating from initial_elo_map is looked up and that value is written into the Elo column setting the Elo rating for each fixture in 2023 to the initial elo rating

    df["HomeElo"] = home_side["Elo"].values # The Elo ratings of the home side before each fixture with 2023 all set to initial ratings are added to the main dataframe
    df["AwayElo"] = away_side["Elo"].values # The Elo ratings of the away side before each fixture with 2023 all set to initial ratings are added to the main dataframe

    season_2023 = df[df["Season"] == 2023].sort_values(["DateParsed", "HomeTeam", "AwayTeam"]) # The main dataframe is filtered for matches falling under the 2023 season and then sorted by Match Date, Home Team and Away Team
    if not season_2023.empty: # Checks if there are matches in the 2023 season if there are no matches the if statement is skipped
        current_ratings = initial_elo_map.copy() # The elo ratings are initialised with the initial elo ratings
        hfa_value = 48.8 # The Home Field Advantage value is set
        k_factor = 20 # The constant which controls how fast the Elo changes is set
        for idx, row in season_2023.iterrows(): # Iterates over each match in the 2023 season in chronological order
            home_team = row["HomeTeam"] # The home team is set
            away_team = row["AwayTeam"] # The away team is set
            home_rating = current_ratings.get(home_team, pd.NA) # The home teams initial Elo rating is looked up and if it is not found it is set to null
            away_rating = current_ratings.get(away_team, pd.NA) # The away teams initial Elo rating is looked up and if it is not found it is set to null
            df.at[idx, "HomeElo"] = home_rating # The initial Elo rating for the home side before the match is stored in df
            df.at[idx, "AwayElo"] = away_rating # The initial Elo rating for the away side before the match is stored in df
            if pd.isna(home_rating) or pd.isna(away_rating): # If the initial elo rating for the home side or away side is null
                continue # Jump to the next match
            home_rating_adjusted = home_rating + hfa_value # The elo rating for the home side is adjusted to include the home field advantage
            expected_home = 1 / (1 + 10 ** (-(home_rating_adjusted - away_rating) / 400)) # The home win probability is computed
            result = row["FullTimeResult"] # The full time result is stored
            if result == "H": # If the home side won
                score_home = 1 # The score result is 1
            elif result == "D": # If the home side drew
                score_home = 0.5 # The score result is 0.5
            else: # If the home side lost
                score_home = 0 # The score result is 0
            goal_margin = abs(row["FullTimeHomeGoals"] - row["FullTimeAwayGoals"]) # The goal difference is calculated
            margin_factor = goal_margin ** 0.5 if goal_margin > 0 else 1 # The goal difference is adjusted to cater for extreme scorelines
            delta_home = k_factor * (score_home - expected_home) * margin_factor # The elo change is calculated
            current_ratings[home_team] = home_rating + delta_home # The new Elo rating for the home side is calculated
            current_ratings[away_team] = away_rating - delta_home # The new elo rating for the away side is calculated

    df["EloTierHome"] = 4 # Sets the Elo tier for the home side in all fixtures to 4
    df.loc[df["HomeElo"] >= 1700, "EloTierHome"] = 3 # If the Elo rating for the home side before that fixture is greater than or equal to 1700 the tier is set to 3
    df.loc[df["HomeElo"] >= 1800, "EloTierHome"] = 2 # If the Elo rating for the home side before that fixture is greater than or equal to 1800 the tier is set to 2
    df.loc[df["HomeElo"] >= 1900, "EloTierHome"] = 1 # If the Elo rating for the home side before that fixture is greater than or equal to 1900 the tier is set to 1
    df["EloTierAway"] = 4 # Sets the Elo tier for the away side in all fixtures to 4
    df.loc[df["AwayElo"] >= 1700, "EloTierAway"] = 3 # If the Elo rating for the away side before that fixture is greater than or equal to 1700 the tier is set to 3
    df.loc[df["AwayElo"] >= 1800, "EloTierAway"] = 2 # If the Elo rating for the away side before that fixture is greater than or equal to 1800 the tier is set to 2
    df.loc[df["AwayElo"] >= 1900, "EloTierAway"] = 1 # If the Elo rating for the away side before that fixture is greater than or equal to 1900 the tier is set to 1

    ratings_cols = [ # Columns to store in form csv file are selected
        "Season", "FifaVersion", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "HomeFifaOverall", "HomeFifaAttack", "HomeFifaMidfield", "HomeFifaDefence",
        "AwayFifaOverall", "AwayFifaAttack", "AwayFifaMidfield", "AwayFifaDefence",
        "HFA", "HomeElo", "AwayElo", 
        "EloTierHome", "EloTierAway",
    ]

    df_ratings = df[ratings_cols].fillna(0) # Null values are replaced with 0
    save_csv(df_ratings, "data/features/eng1_data_ratings.csv") # Csv without null values is saved into the specified directory

    # Combined dataset
    modelling_cols = [
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "Bet365HomeWinOddsPercentage", "Bet365DrawOddsPercentage", "Bet365AwayWinOddsPercentage",
        "OddsDifference_HvA", "OddsDifference_HvD", "OddsDifference_AvD",
        "HomeForm", "AwayForm", "HomeAdvantageIndex",
        "HomeGeneralForm", "AwayGeneralForm", "GeneralFormDifference",
        "AverageGoalsScoredAtHome", "AverageGoalsScoredAtAway",
        "AverageGoalsConcededAtHome", "AverageGoalsConcededAtAway",
        "TotalGoalsScoredHome", "TotalGoalsScoredAway",
        "TotalGoalsConcededHome", "TotalGoalsConcededAway",
        "WinStreakHome", "WinStreakAway",
        "LossStreakHome", "LossStreakAway",
        "TotalWinsHome", "TotalWinsAway",
        "TotalDrawsHome", "TotalDrawsAway",
        "TotalLossesHome", "TotalLossesAway",
        "HistoricalEncountersHome", "HistoricalEncountersAway",
        "HomeFifaOverall", "HomeFifaAttack", "HomeFifaMidfield", "HomeFifaDefence",
        "AwayFifaOverall", "AwayFifaAttack", "AwayFifaMidfield", "AwayFifaDefence",
        "HFA", "HomeElo", "AwayElo",
        "EloTierHome", "EloTierAway",
    ]
    df_combined = df[modelling_cols].fillna(0) # Sets missing values to 0 since form cannot be calculated on first appearance
    save_csv(df_combined, "data/features/eng1_data_combined.csv") # Csv is saved into specified directory

if __name__ == "__main__":
    prepare_features()
