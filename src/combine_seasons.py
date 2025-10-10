import pandas as pd
import glob
from src.utils import ensure_dirs, save_csv

def combine_season_files():
    ensure_dirs("data/raw", "data/processed")

    csv_files = glob.glob("data/raw/Season *.csv")
    all_data = []

    keep_cols = [
        "Date", "HomeTeam", "AwayTeam",
        "FullTimeHomeGoals", "FullTimeAwayGoals", "FullTimeResult",
        "HomeTeamShots", "AwayTeamShots",
        "HomeTeamShotsonTarget", "AwayTeamShotsonTarget",
        "HomeTeamFoulsComitted", "AwayTeamFoulsComitted",
        "HomeTeamCorners", "AwayTeamCorners",
        "HomeTeamYellowCards", "AwayTeamYellowCards",
        "Bet365HomeWinOdds", "Bet365DrawOdds", "Bet365AwayWinOdds"
    ]

    for file in csv_files:
        season_year = file.split("Season ")[1].split(".csv")[0]
        df = pd.read_csv(file)
        df["Season"] = int(season_year)
        df = df[keep_cols + ["Season"]]
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    save_csv(combined, "data/processed/eng1_all_seasons.csv")
    return combined
