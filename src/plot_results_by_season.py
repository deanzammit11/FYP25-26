import pandas as pd
import matplotlib.pyplot as plt
from src.utils import ensure_dirs

def plot_results_by_season(input_csv: str = "data/processed/eng1_all_seasons.csv", output_png: str = "data/results/results_by_season.png", output_csv: str = "data/results/results_by_season.csv"):
    ensure_dirs("data/results") # Checks if directory exists and if it does not it creates it

    df = pd.read_csv(input_csv) # Loads the dataset containing all the seasons combined
    df = df.dropna(subset=["FullTimeResult", "Season"]) # Rows whose FullTimeResult and Season are null are dropped

    count = (df.groupby(["Season", "FullTimeResult"]) # Data is grouped based on each unique Season and FullTimeResult combination
              .size() # Each group occurence is counted
              .unstack(fill_value=0) # The values in the FullTimeResult column are turned into columns and null values are replaced with 0
              .rename(columns={"H": "Home Win", "D": "Draw", "A": "Away Win"}) # The columns are renamed
              .sort_index() # Rows are sorted based by Season
    )
    count.to_csv(output_csv) # Total occurence for each result is saved in results_by_season_count.csv

    ax = count.plot(kind="bar", figsize=(12, 6)) # A grouped bar chart is created using the data from count
    ax.set_title("Match Outcomes by Season (Home/Away/Draw)") # Title for bar chart is set
    ax.set_xlabel("Season") # X-axis label is set
    ax.set_ylabel("Number of Matches") # Y-axis label is set
    ax.legend(title="Outcome, Source", bbox_to_anchor=(1.02, 1), loc="upper left") # Title for legend is set
    plt.tight_layout() # Layout is adjusted
    plt.savefig(output_png, dpi=200) # Figure is saved to the specified path and resolution is set
    plt.close() # Figure is closed

if __name__ == "__main__":
    plot_results_by_season()
