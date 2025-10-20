import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from src.models.evaluate_models import evaluate_model, save_results

def run_xgboost(data_path = "data/features/eng1_data_combined.csv"):
    df = pd.read_csv(data_path) # Loads dataset

    train_df = df[df["Season"] < 2023] # Splits training data using 2019 - 2022 Seasons
    test_df = df[df["Season"] == 2023] # Splits testing data using 2023 Season only

    # Features used to predict are defined
    features = [
        "Bet365HomeWinOddsPercentage",
        "Bet365DrawOddsPercentage",
        "Bet365AwayWinOddsPercentage",
        "OddsDifference_HvA",
        "OddsDifference_HvD",
        "OddsDifference_AvD"
    ]

    X_train = train_df[features] # Features used for training
    y_train = train_df["ResultEncoded"] + 1 # Results used for training and adding 1 since xgboost requires match outcomes to be in the form of 0, 1 and 2 not -1, 0 and 1
    X_test = test_df[features] # Features used for testing
    y_test = test_df["ResultEncoded"] + 1 # Results used for testing and adding 1 since xgboost requires match outcomes to be in the form of 0, 1 and 2 not -1, 0 and 1
    
    param_grid = {
        "learning_rate": [0.05, 0.1, 0.2], # Step size at each boosting step
        "n_estimators": [300, 400, 500], # Number of trees to build
        "max_depth": [4, 6, 8], # Maximum depth of each tree
        "min_child_weight": [1, 5], # Minimum sum of instance weights in a child node
        "gamma": [0, 1], # Minimum loss reduction required to further split a leaf node
        "subsample": [0.8, 1.0], # Fraction of training samples used for each boosting round
        "colsample_bytree": [0.8, 1.0], # Fraction of features used by tree
        "reg_lambda": [0.5, 1.0, 2.0], # L2 regularisation term
        "reg_alpha": [0.0, 0.5], # L1 regularisation term
        "scale_pos_weight": [1, 1.5], # Adjusts for class imbalance where 1 = no correction and 1.5 = slightly favors minority classes
        "grow_policy": ["depthwise"] # Tree growth strategy
    }

    base_model = XGBClassifier(
        objective = "multi:softprob", # Multi-class classification outputting class probabilities.
        num_class = 3, # Refers to 3 possible outcome classes
        eval_metric = "mlogloss", # Measures how well predicted probabilities match true labels.
        random_state = 0, # Ensure results are reproducible and consistent
        n_jobs = -1, # Enables usage of all available CPU cores
        tree_method = "hist", # Uses a histogram based algorithm for building the trees
    )

    grid_search = GridSearchCV( # Grid search is setup to find the best combination of parameters through cross validation
        estimator = base_model, # The model which is being optmised
        param_grid = param_grid, # The parameter combinations to test
        scoring = "accuracy", # The metric used to evaluate performance
        cv = 3, # Cross validation is set to 3 fold
        n_jobs = -1, # All CPU cores are utilised
        verbose = 2, # Progress is displayed in terminal
        error_score = "raise" # Error is raised if a fit fails
    )

    print("Running grid search.") # Prints message to reassure that grid search is running
    grid_search.fit(X_train, y_train) # Parameter tuning is performed

    print("Grid Search Complete. Best Parameters:") # Prints message confirmation message that grid search completed successfully
    print(grid_search.best_params_) # Best parameters found from grid search are printed
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}") # Best cross validation accuracy is printed

    model = grid_search.best_estimator_ # Model trained using the best parameters is retrieved
    preds = model.predict(X_test) # Predicted outcomes are stored
    preds = preds - 1 # Subtracting 1 to convert predicted outcomes back to the original format
    y_test = y_test - 1 # Subtracting 1 to convert testing outcomes back to the original format

    results = evaluate_model("XGBoost", y_test, preds) # Model performance is evaluated and stored in results
    save_results(results) # Model results are saved

    out = test_df.copy() # It stored a copy of the dataframe used for testing in out
    out["Predicted"] = preds # Add a predicted column to the dataframe
    out.to_csv("data/results/xgboost_2023_predictions.csv", index = False) # Converts the final dataframe to csv and appends it to existing csv or stores it in new csv
    print("Predictions saved to: data/results/xgboost_2023_predictions.csv") # Prints confirmation that the results have been stored
    return model, results # Returns the trained model and the respective results

if __name__ == "__main__":
    run_xgboost()