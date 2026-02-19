import pandas as pd
from src.utils import ensure_dirs, save_csv

raw_fifa_teams_path = "data/raw/fifa 15 - fc 24 teams data.csv" # Stores the path of fifa ratings dataset
output_path = "data/processed/fifa_20-24_teams_data.csv" # Stores the path where the filtered dataset will be saved
target_league_id = 13 # Stores the League id for the Premier League
target_versions = {20, 21, 22, 23, 24} # Stores the fifa versions to filter for

def filter_fifa_teams(source_path: str = raw_fifa_teams_path, output_path: str = output_path):
    ensure_dirs("data/processed") # Checks if directory exists and if it does not it creates it

    df = pd.read_csv(source_path) # Fifa ratings csv file is loaded into a Dataframe

    required_columns = {"league_id", "fifa_version"} # The columns which must be present in the dataset are defined
    missing_columns = required_columns.difference(df.columns) # Missing columns are stored
    if missing_columns: # If there are any missing columns
        raise ValueError(f"Missing expected columns: {', '.join(sorted(missing_columns))}") # An error showing which columns are msising is shown and it stops executing

    df["league_id"] = pd.to_numeric(df["league_id"], errors="coerce") # The values in the league_id column are converted to numeric values with invalid values being null
    df["fifa_version"] = pd.to_numeric(df["fifa_version"], errors="coerce") # The values in the fifa_version column are converted to numeric values with invalid values being null

    filter = (df["league_id"].eq(target_league_id) & df["fifa_version"].isin(target_versions)) # A filtering condition where the league_id must be 13 and fifa version must be between 20 and 24 is defined
    filtered = df.loc[filter].copy() # Dataframe is filtered using the defined filter

    save_csv(filtered, output_path) # New dataframe is saved as a csv file in the define output path
    return filtered # The filtered dataframe is returned

if __name__ == "__main__":
    filter_fifa_teams()
