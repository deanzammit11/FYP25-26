import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from src.utils import ensure_dirs, save_csv

def score_outcomes_by_season():
    df = pd.read_csv("data/processed/eng1_all_seasons.csv") # Csv file is read from the specified directory and stored in a dataframe

    required_columns = {"Season", "FullTimeHomeGoals", "FullTimeAwayGoals"} # Required columns for score outcome counting are defined
    missing = required_columns.difference(df.columns) # Any missing required columns are identified
    if missing: # If one or more required columns are missing
        raise ValueError(f"Missing required columns: {sorted(missing)}") # A ValueError is raised

    score = df.dropna(subset=["Season", "FullTimeHomeGoals", "FullTimeAwayGoals"]).copy() # Rows missing season or full-time score are removed and the final result is copied into a dataframe
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

    score["Margin"] = (score["FullTimeHomeGoals"] - score["FullTimeAwayGoals"]).abs().astype(int) # Absolute goal margin is computed and stored as an integer

    margin_count = score["Margin"].value_counts().sort_index() # Margin frequencies are counted and sorted by margin in ascending order
    margin_prob = score["Margin"].value_counts(normalize=True).sort_index() # Margin probabilities are computed and sorted by margin in ascending order

    fig, ax = plt.subplots(figsize=(10, 5)) # A matplotlib figure and axis are created for the bar chart image with the width being set to 10.5
    ax.bar(margin_count.index.astype(str), margin_count.values, color="#4c78a8") # Margin counts are plotted with the x-axis having margin labels while the y-axis having the counts for each margin
    ax.set_title("Goal Margin Distribution") # The title for the margin chart is set
    ax.set_xlabel("Absolute Goal Margin") # The x-axis label for the margin chart is set
    ax.set_ylabel("Number of Matches") # The y-axis label for the margin chart is set
    ax.grid(axis="y", alpha=0.2) # Grid lines of transparency 0.2 are added across the y-axis
    if not margin_prob.empty: # If at least one margin value exists
        most_common_margin = int(margin_prob.idxmax()) # The most common margin is stored
        most_common_prob = float(margin_prob.loc[most_common_margin]) # The probability of the most common margin is stored
        ax.text(0.98, 0.95, f"Most common margin: {most_common_margin} ({most_common_prob:.1%})", transform=ax.transAxes, ha="right", va="top") # A comment showing the common margin and its percentage correct to 1 decimal place is added
    plt.tight_layout() # Layout is adjusted
    plt.savefig("data/results/margin_distribution.png", dpi=200) # Image is saved to the specified path setting the resolution
    plt.close() # Figure is closed

    lambda_home = float(score["FullTimeHomeGoals"].mean()) # The mean home goal count acting as the poisson lambda is computed
    lambda_away = float(score["FullTimeAwayGoals"].mean()) # The mean away goal count acting as the poisson lambda is computed
    max_goal = int(max(score["FullTimeHomeGoals"].max(), score["FullTimeAwayGoals"].max())) # The most goals scored by any side is stored
    k_values = np.arange(0, max_goal + 1) # An array of goal counts starting from 0 up to max_goal inclusive is created

    observed_home = score["FullTimeHomeGoals"].value_counts(normalize=True).sort_index() # The probability of each home goal count is computed and is ordered in ascending order
    observed_away = score["FullTimeAwayGoals"].value_counts(normalize=True).sort_index() # The probability of each away goal count is computed and is ordered in ascending order
    observed_home_probs = [float(observed_home.get(k, 0.0)) for k in k_values] # For each goal count in the range the probability that the home side scores k goals is added from observed_home or set to 0 if that number of goals were never scored
    observed_away_probs = [float(observed_away.get(k, 0.0)) for k in k_values] # For each goal count in the range the probability that the away side scores k goals is added from observed_away or set to 0 if that number of goals were never scored

    poisson_home_probs = [math.exp(-lambda_home) * (lambda_home ** int(k)) / math.factorial(int(k)) for k in k_values] # For each goal count k the poisson probability mass function for home goals is computed
    poisson_away_probs = [math.exp(-lambda_away) * (lambda_away ** int(k)) / math.factorial(int(k)) for k in k_values] # For each goal count k the poisson probability mass function for away goals is computed

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True) # A matplotlib figure and two axes are created for side by side plots with the width being set to 14 and the y axis being shared

    axes[0].plot(k_values, observed_home_probs, "o-", label="Observed", color="#1f77b4") # The observed home goal probabilities are plotted using circle markers and a solid line with the legend label being set to Observed
    axes[0].plot(k_values, poisson_home_probs, "s--", label=f"Poisson (lambda={lambda_home:.2f})", color="#ff7f0e") # The Fitted Poisson curve for home score is plotted using square markers and a dashed line with the legend label being set to Poisson and the lambda value in brackets
    axes[0].set_title("Home score: Observed vs Poisson") # The title for the home score plot is set
    axes[0].set_xlabel("score (k)") # The x-axis label for the home score plot is set
    axes[0].set_ylabel("Probability") # The y-axis label for the home score plot is set
    axes[0].legend() # Legend is displayed
    axes[0].grid(alpha=0.2) # Grid lines of transparency 0.2 are added across the y-axis

    axes[1].plot(k_values, observed_away_probs, "o-", label="Observed", color="#2ca02c") # The observed away goal probabilities are plotted using circle markers and a solid line with the legend label being set to Observed
    axes[1].plot(k_values, poisson_away_probs, "s--", label=f"Poisson (lambda={lambda_away:.2f})", color="#d62728") # The Fitted Poisson curve for away score is plotted using square markers and a dashed line with the legend label being set to Poisson and the lambda value in brackets
    axes[1].set_title("Away score: Observed vs Poisson") # The title for the away score plot is set
    axes[1].set_xlabel("score (k)") # The x-axis label for the away score plot is set
    axes[1].legend() # Legend is displayed
    axes[1].grid(alpha=0.2) # Grid lines of transparency 0.2 are added across the y-axis

    fig.suptitle("Poisson Fit to Goal Distributions", fontsize=12) # The figure title is set
    plt.tight_layout() # Layout is adjusted
    plt.savefig("data/results/poisson_goals_distribution.png", dpi=200) # Image is saved to the specified path setting the resolution
    plt.close() # Figure is closed
    print(f"Saved Poisson fit to goal distributions plot: {"data/results/score_outcomes_by_elo_tier.png"}") # A confirmation message showing where the plot was saved is printed

if __name__ == "__main__":
    score_outcomes_by_season()