import pandas as pd
from src.utils import ensure_dirs, save_csv

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

    home_results = df[["Date", "HomeTeam", "HomePoints"]].rename(columns={"HomeTeam": "Team", "HomePoints": "Points"}) # A table storing the the points scored by the home team per fixture is created
    away_results = df[["Date", "AwayTeam", "AwayPoints"]].rename(columns={"AwayTeam": "Team", "AwayPoints": "Points"}) # A table storing the the points scored by the away team per fixture is created

    all_results = pd.concat([home_results, away_results]).sort_values(["Team", "Date"]) # Both tables are joined into one table and sorted by team name and then by date

    all_results["GeneralForm"] = (
        all_results.groupby("Team")["Points"].shift() # Points scored are grouped by team and shifted one game back
        .rolling(window=5, min_periods=1).sum() # Sum of points scored over the last 5 games is computed
        .reset_index(level=0, drop=True) # The Team grouping in the multi level index is dropped leaving only the row id
    )

    df = df.merge(
        all_results[["Team", "Date", "GeneralForm"]], # Merge df with the Team, Date and GeneralForm columns in all_results
        how="left", # Keeps all columns in df and only merges the columns from all_results
        left_on=["HomeTeam", "Date"], right_on=["Team", "Date"] # Match HomeTeam and Date from the left table and Team and Date from the right table and if they match merge
    ).rename(columns={"GeneralForm": "HomeGeneralForm"}).drop(columns="Team") # GeneralForm column is renamed and the new Team column is removed

    df = df.merge(
        all_results[["Team", "Date", "GeneralForm"]], # Merge df with the Team, Date and GeneralForm columns in all_results
        how="left", # Keeps all columns in df and only merges the columns from all_results
        left_on=["AwayTeam", "Date"], right_on=["Team", "Date"] # Match AwayTeam and Date from the left table and Team and Date from the right table and if they match merge
    ).rename(columns={"GeneralForm": "AwayGeneralForm"}).drop(columns="Team") # GeneralForm column is renamed and the new Team column is removed

    df["GeneralFormDifference"] = df["HomeGeneralForm"] - df["AwayGeneralForm"] # Points scored by the away team in the last 5 matches are subtracted from the points scored by the home team in the last 5 matches

    form_cols = [ # Columns to store in form csv file are selected
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "HomeForm", "AwayForm", "HomeAdvantageIndex",
        "HomeGeneralForm", "AwayGeneralForm", "GeneralFormDifference"
    ]

    df_form = df[form_cols].dropna() # Null values are removed
    save_csv(df_form, "data/features/eng1_data_form.csv") # Csv without null values is saved into the specified directory

    # Combined dataset
    modelling_cols = [
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "Bet365HomeWinOddsPercentage", "Bet365DrawOddsPercentage", "Bet365AwayWinOddsPercentage",
        "OddsDifference_HvA", "OddsDifference_HvD", "OddsDifference_AvD",
        "HomeForm", "AwayForm", "HomeAdvantageIndex",
        "HomeGeneralForm", "AwayGeneralForm", "GeneralFormDifference"
    ]
    df_combined = df[modelling_cols].dropna() # Null values are removed
    save_csv(df_combined, "data/features/eng1_data_combined.csv") # Csv without null values is saved into specified directory

if __name__ == "__main__":
    prepare_features()
