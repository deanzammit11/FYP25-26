import math
import matplotlib.pyplot as plt
import pandas as pd
from src.utils import ensure_dirs, save_csv

def score_outcomes_by_elo_tier():
    match_df = pd.read_csv("data/processed/eng1_all_seasons.csv") # Csv file is read from the specified directory and stored in a dataframe
    features_df = pd.read_csv("data/features/eng1_data_combined.csv") # Csv file is read from the specified directory and stored in a dataframe

    required_match_columns = {"Season", "Date", "HomeTeam", "AwayTeam", "FullTimeHomeGoals", "FullTimeAwayGoals"} # The required columns from the match results dataframe are defined
    missing_match_columns = required_match_columns.difference(match_df.columns) # Any missing required columns are identified
    if missing_match_columns: # If one or more required columns are missing
        raise ValueError(f"Missing required columns in {"data/processed/eng1_all_seasons.csv"}: {missing_match_columns}") # A ValueError is raised

    required_feature_columns = {"Season", "Date", "HomeTeam", "AwayTeam", "EloTierHome", "EloTierAway"} # The required columns from the features dataframe are defined
    missing_feature_columns = required_feature_columns.difference(features_df.columns) # Any missing required columns are identified
    if missing_feature_columns: # If one or more required feature columns are missing
        raise ValueError(f"Missing required columns in {"data/features/eng1_data_combined.csv"}: {missing_feature_columns}") # A ValueError is raised

    match_df = match_df[list(required_match_columns)].copy() # Only the required match result columns are kept and the filtered dataframe is copied
    features_df = features_df[list(required_feature_columns)].copy() # Only the required feature columns are kept and the filtered dataframe is copied

    match_df["Season"] = pd.to_numeric(match_df["Season"], errors="coerce") # The Season column in the match dataframe is converted to numeric with any invalid values becoming null
    features_df["Season"] = pd.to_numeric(features_df["Season"], errors="coerce") # The Season column in the features dataframe is converted to numeric with any invalid values becoming null
    match_df["FullTimeHomeGoals"] = pd.to_numeric(match_df["FullTimeHomeGoals"], errors="coerce") # The home goals column in the match dataframe is converted to numeric with any invalid values becoming null
    match_df["FullTimeAwayGoals"] = pd.to_numeric(match_df["FullTimeAwayGoals"], errors="coerce") # The away goals column in the match dataframe is converted to numeric with any invalid values becoming null
    features_df["EloTierHome"] = pd.to_numeric(features_df["EloTierHome"], errors="coerce") # The home Elo tier column in the features dataframe is converted to numeric with any invalid values becoming null
    features_df["EloTierAway"] = pd.to_numeric(features_df["EloTierAway"], errors="coerce") # The away Elo tier column in the features dataframe is converted to numeric with any invalid values becoming null

    merged = match_df.merge(features_df, how="inner", on=["Season", "Date", "HomeTeam", "AwayTeam"]) # The match results and engineered features dataframes are joined together combining rows which have the same Season, Date, HomeTeam and AwayTeam
    merged = merged.dropna(subset=["Season", "FullTimeHomeGoals", "FullTimeAwayGoals", "EloTierHome", "EloTierAway"]).copy() # Rows where the season, home goals, away goals, home elo tier or away elo tier are null are removed and the filtered dataframe is then copied
    if merged.empty: # If no rows are left after the join and the filtering
        raise ValueError("No rows remained after joining match results to Elo tier features.") # A ValueError is raised

    merged["Season"] = merged["Season"].astype(int) # The Season column is converted to integer
    merged["FullTimeHomeGoals"] = merged["FullTimeHomeGoals"].astype(int) # The home goals column is converted to integer
    merged["FullTimeAwayGoals"] = merged["FullTimeAwayGoals"].astype(int) # The away goals column is converted to integer
    merged["EloTierHome"] = merged["EloTierHome"].astype(int) # The home Elo tier column is converted to integer
    merged["EloTierAway"] = merged["EloTierAway"].astype(int) # The away Elo tier column is converted to integer
    merged["GoalMargin"] = (merged["FullTimeHomeGoals"] - merged["FullTimeAwayGoals"]).abs().astype(int) # The absolute goal margin is computed for each match and stored as an integer
    merged["Matchup"] = (merged["EloTierHome"].astype(str) + " vs " + merged["EloTierAway"].astype(str)) # A matchup label is built in the format home tier vs away tier

    result = ( # A dataframe which counts the margins for each Elo tier matchup is built
        merged.groupby(["EloTierHome", "EloTierAway", "Matchup", "GoalMargin"], as_index=False) # Rows are grouped by home tier, away tier, matchup and absolute goal margin
        .size() # The number of rows in each group are counted
        .rename(columns={"size": "Count"}) # The resulting size column is renamed to Count
        .sort_values(["EloTierHome", "EloTierAway", "GoalMargin"]) # The grouped rows are sorted by home tier, away tier and then goal margin
        .reset_index(drop=True) # The row index is reset
    )

    ensure_dirs("data/results") # Checks if directory exists and if it does not it creates it
    save_csv(result, "data/results/score_outcomes_by_elo_tier.csv") # Csv is saved into specified directory

    matchup_order = ( # A dataframe which stores the list of unique Elo tier matchups to define subplot order is built
        result[["EloTierHome", "EloTierAway", "Matchup"]] # The required columns are selected
        .drop_duplicates() # Duplicate matchup rows are removed so each matchup appears once
        .sort_values(["EloTierHome", "EloTierAway"]) # Matchups are sorted by home tier and then by away tier
        .reset_index(drop=True) # The row index is reset
    )

    num_matchups = len(matchup_order) # The number of unique elo tier matchups are counted
    num_cols = min(4, num_matchups) # The subplot grid is set to the smallest value between 4 and the number of unique elo tier matchups
    num_rows = math.ceil(num_matchups / num_cols) # The number of rows required to display all matchup plots is computed
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False) # A subplot grid containing num_rows of height 4, num_cols of width 5 is created
    axes_flat = axes.flatten() # The subplot axes are flattened into a one dimensional array

    for ax, matchup_row in zip(axes_flat, matchup_order.itertuples(index=False)): # For each subplot axis which is paired with one directed Elo tier matchup
        matchup_df = result[(result["EloTierHome"] == matchup_row.EloTierHome) & (result["EloTierAway"] == matchup_row.EloTierAway)].copy() # The result dataframe is filtered for rows where the matchup matches the current matchup and the final dataframe is copied
        matchup_df = matchup_df.sort_values("GoalMargin") # The matchup rows are sorted by goal margin in ascending order so the bars appear in order

        ax.bar(matchup_df["GoalMargin"].astype(str), matchup_df["Count"], color="#4c78a8") # A blue bar chart with the goal margins on the x-axis, the number of matches on the y-axis is plotted
        ax.set_title(f"{matchup_row.Matchup} (n={int(matchup_df['Count'].sum())})") # The subplot title showing the elo tier matchup and the total number of matches in that matchup is set
        ax.set_xlabel("Absolute Goal Margin") # The x-axis label is set
        ax.set_ylabel("Number of Matches") # The y-axis label is set
        ax.grid(axis="y", alpha=0.2) # Grid lines of transparency 0.2 are added across the y-axis

    for ax in axes_flat[num_matchups:]: # For each subplot axes which was not used after all matchups have been plotted
        ax.axis("off") # The subplot is hidden

    fig.suptitle("Goal Margins by Elo Tier Matchup", fontsize=14) # The figure title is set
    plt.tight_layout() # Layout is adjusted
    plt.savefig("data/results/score_outcomes_by_elo_tier.png", dpi=200, bbox_inches="tight") # Image is saved to the specified path setting the resolution
    plt.close() # Figure is closed
    print(f"Saved Elo tier goal margin plot: {"data/results/score_outcomes_by_elo_tier.png"}") # A confirmation message showing where the plot was saved is printed

if __name__ == "__main__":
    score_outcomes_by_elo_tier()