import pandas as pd
from src.utils import ensure_dirs, save_csv

def prepare_features():
    ensure_dirs("data/features") # Checks if directory exists and if it does not it creates it
    df = pd.read_csv("data/processed/eng1_all_seasons.csv") # Csv file is read from the specified directory and stored in a data frame

    df = df.dropna(subset=["FullTimeResult"]) # Drop rows with missing results

    df["ResultEncoded"] = df["FullTimeResult"].map({"H": 1, "D": 0, "A": -1}) # Encode target
    
    # Betting Odds Features

    # Bet365 odds are converted into implied probabilities
    df["Home"] = 1 / df["Bet365HomeWinOdds"]
    df["Draw"] = 1 / df["Bet365DrawOdds"]
    df["Away"] = 1 / df["Bet365AwayWinOdds"]

    df["Total"] = df["Home"] + df["Draw"] + df["Away"] # Calculate the total of the implied probabilities

    df["Bet365HomeWinOddsPercentage"] = df["Home"] / df["Total"] # Convert the implied home win probability into a percentage
    df["Bet365DrawOddsPercentage"] = df["Draw"] / df["Total"] # Convert the implied draw probability into a percentage
    df["Bet365AwayWinOddsPercentage"] = df["Away"] / df["Total"] # Convert the implied away win probability into a percentage

    #df["OddsDifference_HvA"] = (df["Bet365HomeWinOddsPercentage"] - df["Bet365AwayWinOddsPercentage"]) # Home vs away odds difference without absolute value
    #df["OddsDifference_HvD"] = (df["Bet365HomeWinOddsPercentage"] - df["Bet365DrawOddsPercentage"]) # Home vs draw odds difference without absolute value
    #df["OddsDifference_AvD"] = (df["Bet365AwayWinOddsPercentage"] - df["Bet365DrawOddsPercentage"]) # Away vs draw odds difference without absolute value
    df["OddsDifference_HvA"] = (df["Bet365HomeWinOddsPercentage"] - df["Bet365AwayWinOddsPercentage"]).abs() # Home vs away odds difference with absolute value
    df["OddsDifference_HvD"] = (df["Bet365HomeWinOddsPercentage"] - df["Bet365DrawOddsPercentage"]).abs() # Home vs draw odds difference with absolute value
    df["OddsDifference_AvD"] = (df["Bet365AwayWinOddsPercentage"] - df["Bet365DrawOddsPercentage"]).abs() # Away vs draw odds difference with absolute value

    odds_cols = [ # Columns to store in odds csv file are selected
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "Bet365HomeWinOddsPercentage", "Bet365DrawOddsPercentage", "Bet365AwayWinOddsPercentage",
        "OddsDifference_HvA", "OddsDifference_HvD", "OddsDifference_AvD"
    ]
    df_odds = df[odds_cols].dropna() # Null values are removed
    save_csv(df_odds, "data/features/eng1_data_odds.csv") # Csv without null values is saved into the specified directory

    # Home Advantage Features
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce") # The data column strings are converted into datetime format with the day being first and an invalid entry being counted as not a time
    df = df.sort_values(["Season", "Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True) # Rows are first sorted by season, then by match date, and finally by the home and away club names and old index is replaced with a new one

    home_points = df["ResultEncoded"].map({1: 3, 0: 1, -1: 0}) # Result is mapped to points
    away_points = df["ResultEncoded"].map({1: 0, 0: 1, -1: 3}) # Result is mapped to points

    df["HomeAveragePoints"] = (
        home_points.groupby(df["HomeTeam"]).shift() # Each Home team's home results is shifted one game back
        .groupby(df["HomeTeam"]).rolling(window=5, min_periods=1).mean() # Average points scored by the home team over the last 5 home games is computed
        .reset_index(level=0, drop=True) # The HomeTeam grouping in the multi level index is dropped leaving only the row id
    )
    df["AwayAveragePoints"] = (
        away_points.groupby(df["AwayTeam"]).shift() # Each Away team's away results is shifted one game back
        .groupby(df["AwayTeam"]).rolling(window=5, min_periods=1).mean() # Average points scored by the away team over the last 5 away games is computed
        .reset_index(level=0, drop=True) # The AwayTeam grouping in the multi level index is dropped leaving only the row id
    )
    df["HomeAdvantageIndex"] = (df["HomeAveragePoints"] - df["AwayAveragePoints"]).fillna(0.0) # Average away points scored are subtracted from average home points scored and missing values are filled with 0

    home_advantage_cols = [ # Columns to store in home advantage csv file are selected
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "HomeAveragePoints", "AwayAveragePoints", "HomeAdvantageIndex"
    ]
    df_home_advantage = df[home_advantage_cols].dropna() # Null values are removed
    save_csv(df_home_advantage, "data/features/eng1_data_home_advantage.csv") # Csv without null values is saved into the specified directory

    # Combined dataset
    modelling_cols = [
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "Bet365HomeWinOddsPercentage", "Bet365DrawOddsPercentage", "Bet365AwayWinOddsPercentage",
        "OddsDifference_HvA", "OddsDifference_HvD", "OddsDifference_AvD",
        "HomeAdvantageIndex"
    ]
    df_combined = df[modelling_cols].dropna() # Null values are removed
    save_csv(df_combined, "data/features/eng1_data_combined.csv") # Csv without null values is saved into specified directory

if __name__ == "__main__":
    prepare_features()
