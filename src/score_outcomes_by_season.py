import pandas as pd
from src.utils import ensure_dirs, save_csv

def score_outcomes_by_season():
    df = pd.read_csv("data/processed/eng1_all_seasons.csv") # Csv file is read from the specified directory and stored in a dataframe

    required_columns = {"Season", "FullTimeHomeGoals", "FullTimeAwayGoals"} # Required columns for score outcome counting are defined
    missing = required_columns.difference(df.columns) # Any missing required columns are identified
    if missing: # If one or more required columns are missing
        raise ValueError(f"Missing required columns: {sorted(missing)}") # A ValueError is raised

    score = df.dropna(subset=["Season", "FullTimeHomeGoals", "FullTimeAwayGoals"]).copy() # Rows missing season or full-time goals are removed and the final result is copied into a dataframe
    score["Season"] = score["Season"].astype(int) # Season values are converted to integers
    score["FullTimeHomeGoals"] = score["FullTimeHomeGoals"].astype(int) # FullTimeHomeGoals are converted to integers
    score["FullTimeAwayGoals"] = score["FullTimeAwayGoals"].astype(int) # FullTimeAwayGoals are converted to integers
    score["ScoreOutcome"] = (score["FullTimeHomeGoals"].astype(str) + "-" + score["FullTimeAwayGoals"].astype(str)) # A score outcome column contaning the outcome in home-away form as a string is added

    season = (score.groupby(["Season", "ScoreOutcome"], as_index=False).size().rename(columns={"size": "Count"})) # Each Season, ScoreOutcome pair is grouped and every occurence of that group is counted with the size column being renamed to Count

    total = (score.groupby("ScoreOutcome", as_index=False).size().rename(columns={"size": "Count"})) # All matches are grouped by and every unique outcome occurence is counted with the size column being renamed to Count
    total.insert(0, "Season", "Total") # It adds a new column named Season as the first column and fills every value with the string "Total"

    result = pd.concat([season, total], ignore_index=True) # The season and total score outcome counts are concatenated into a single dataframe
    result["Season"] = result["Season"].astype(str) # Season is converted to string
    result = result.sort_values(["ScoreOutcome", "Season"]) # Rows are sorted by ScoreOutcome and then by season

    ensure_dirs("data/results") # Checks if directory exists and if it does not it creates it
    save_csv(result, "data/results/score_outcomes_by_season.csv") # Csv is saved into specified directory
    return result # The final dataframe is returned

if __name__ == "__main__":
    score_outcomes_by_season()
