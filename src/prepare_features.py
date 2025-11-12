import pandas as pd
from src.utils import ensure_dirs, save_csv

def prepare_features():
    ensure_dirs("data/features") # Checks if directory exists and if it does not it creates it
    df = pd.read_csv("data/processed/eng1_all_seasons.csv") # Csv file is read from the specified directory and stored in a data frame

    df = df.dropna(subset=["FullTimeResult"]) # Drop rows with missing results

    df["ResultEncoded"] = df["FullTimeResult"].map({"H": 1, "D": 0, "A": -1}) # Encode target
    
    # Bet365 odds are converted into implied probabilities
    df["Home"] = 1 / df["Bet365HomeWinOdds"]
    df["Draw"] = 1 / df["Bet365DrawOdds"]
    df["Away"] = 1 / df["Bet365AwayWinOdds"]

    df["Total"] = df["Home"] + df["Draw"] + df["Away"] # Calculate the total of the implied probabilities
    df["Bet365HomeWinOddsPercentage"] = df["Home"] / df["Total"] # Convert the implied home win probability into a percentage
    df["Bet365DrawOddsPercentage"] = df["Draw"] / df["Total"] # Convert the implied draw probability into a percentage
    df["Bet365AwayWinOddsPercentage"] = df["Away"] / df["Total"] # Convert the implied away win probability into a percentage

    # Derived features
    df["OddsDifference_HvA"] = df["Bet365HomeWinOddsPercentage"] - df["Bet365AwayWinOddsPercentage"] # Home vs away odds difference
    df["OddsDifference_HvD"] = df["Bet365HomeWinOddsPercentage"] - df["Bet365DrawOddsPercentage"] # Home vs draw odds difference
    df["OddsDifference_AvD"] = df["Bet365AwayWinOddsPercentage"] - df["Bet365DrawOddsPercentage"] # Away vs draw odds difference

    # Bookmaker odds features
    odds_cols = [
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "Bet365HomeWinOddsPercentage", "Bet365DrawOddsPercentage", "Bet365AwayWinOddsPercentage",
        "OddsDifference_HvA", "OddsDifference_HvD", "OddsDifference_AvD"
    ]
    df_odds = df[odds_cols].dropna() # Null values are removed
    save_csv(df_odds, "data/features/eng1_data_odds.csv") # Csv without null values is saved into the specified directory

    # Combined dataset
    modelling_cols = [
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "Bet365HomeWinOddsPercentage", "Bet365DrawOddsPercentage", "Bet365AwayWinOddsPercentage",
        "OddsDifference_HvA", "OddsDifference_HvD", "OddsDifference_AvD"
    ]
    df_combined = df[modelling_cols].dropna() # Null values are removed
    save_csv(df_combined, "data/features/eng1_data_combined.csv") # Csv without null values is saved into specified directory
