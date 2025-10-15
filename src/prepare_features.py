import pandas as pd
from src.utils import ensure_dirs, save_csv

def prepare_features():
    ensure_dirs("data/features")
    df = pd.read_csv("data/processed/eng1_all_seasons.csv")

    df = df.dropna(subset=["FullTimeResult"]) # Drop rows with missing results

    df["ResultEncoded"] = df["FullTimeResult"].map({"H": 1, "D": 0, "A": -1}) # Encode target

    # Derived features
    df["OddsDifference_Bet365"] = df["Bet365AwayWinOdds"] - df["Bet365HomeWinOdds"]

    # Bookmaker odds features
    odds_cols = [
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "Bet365HomeWinOdds", "Bet365DrawOdds", "Bet365AwayWinOdds",
        "OddsDifference_Bet365"
    ]
    df_odds = df[odds_cols].dropna()
    save_csv(df_odds, "data/features/eng1_data_odds.csv")

    # Combined dataset
    modelling_cols = [
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "Bet365HomeWinOdds", "Bet365DrawOdds", "Bet365AwayWinOdds",
        "OddsDifference_Bet365"
    ]
    df_combined = df[modelling_cols].dropna()
    save_csv(df_combined, "data/features/eng1_data_combined.csv")
