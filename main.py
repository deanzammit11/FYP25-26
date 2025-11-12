from src.combine_seasons import combine_season_files
from src.prepare_features import prepare_features
from src.plot_results_by_season import plot_results_by_season
from src.plot_predicted_vs_actual import plot_predicted_vs_actual_by_season

def main():
    print("Combining season CSV files...")
    combine_season_files()

    print("Preparing features for modeling...")
    prepare_features()

    print("Plotting season by season results chart...")
    plot_results_by_season()
    plot_predicted_vs_actual_by_season()

    print("Data preparation complete.")

if __name__ == "__main__":
    main()
