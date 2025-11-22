import pandas as pd
import glob
from src.utils import ensure_dirs, save_csv

def combine_season_files():
    ensure_dirs("data/raw", "data/processed") # Checks if directory exists and if it does not it creates it

    csv_files = glob.glob("data/raw/Season *.csv") # All csv's which match that pattern in that directory are stored
    all_data = [] # List to store season dataframes is initialised

    keep_cols = [ # Columns to keep from each file
        "Date", "HomeTeam", "AwayTeam",
        "FullTimeHomeGoals", "FullTimeAwayGoals", "FullTimeResult",
        "HomeTeamShots", "AwayTeamShots",
        "HomeTeamShotsonTarget", "AwayTeamShotsonTarget",
        "HomeTeamFoulsComitted", "AwayTeamFoulsComitted",
        "HomeTeamCorners", "AwayTeamCorners",
        "HomeTeamYellowCards", "AwayTeamYellowCards",
        "Bet365HomeWinOdds", "Bet365DrawOdds", "Bet365AwayWinOdds"
    ]

    for file in csv_files: # Each season csv file is iterated
        season_year = file.split("Season ")[1].split(".csv")[0] # Season year is obtained from the file name
        df = pd.read_csv(file) # Season csv file is loaded into a Dataframe
        df["Season"] = int(season_year) # An integer column named Season is added containing the same values for all rows of the same file
        df = df[keep_cols + ["Season"]] # Only the rows to keep are selected and they are appended with the new Season column
        all_data.append(df) # The dataframe for that season is appended to the list

    combined = pd.concat(all_data, ignore_index=True) # The rows of each dataframe in all_data are appended maintaining the same order
    save_csv(combined, "data/processed/eng1_all_seasons.csv") # Combined csv is saved into specified directory
    return combined # The combined dataframe is returned

if __name__ == "__main__":
    combine_season_files()
