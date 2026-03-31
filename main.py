from src.combine_seasons import combine_season_files
from src.filter_fifa_teams import filter_fifa_teams
from src.prepare_features import prepare_features
from src.plot_results_by_season import plot_results_by_season
from src.score_outcomes_by_season import score_outcomes_by_season
from src.score_outcomes_by_elo_tier import score_outcomes_by_elo_tier
from src.models.logistic_regression import run_logistic_regression
from src.models.random_forest import run_random_forest
from src.models.xgboost import run_xgboost
from src.plot_predicted_vs_actual import plot_predicted_vs_actual_by_season
from src.season_simulation import season_simulation
from src.betting_simulation import betting_simulation
from src.pca_feature_analysis import pca_feature_analysis
from src.feature_importance_analysis import feature_importance_analysis

def main():
    print("Combining season CSV files...")
    combine_season_files()

    print("Filtering FIFA teams data...")
    filter_fifa_teams()

    print("Preparing features for modelling...")
    prepare_features()

    print("Plotting season by season results chart...")
    plot_results_by_season()

    print("Plotting Observed vs Poisson goal distribution...")
    score_outcomes_by_season()

    print("Plotting goal margins by Elo tier matchups charts...")
    score_outcomes_by_elo_tier()

    print("Running logistic regression model...")
    run_logistic_regression()

    print("Running random forest model...")
    run_random_forest()

    print("Running XGBoost model...")
    run_xgboost()

    print("Plotting predicted vs actual outcomes...")
    plot_predicted_vs_actual_by_season()

    print("Simulating the 2023 season table...")
    season_simulation()

    print("Running betting simulation...")
    betting_simulation()

    print("Running PCA feature analysis...")
    pca_feature_analysis()

    print("Running feature importance analysis...")
    feature_importance_analysis()

    print("Pipeline complete.")

if __name__ == "__main__":
    main()