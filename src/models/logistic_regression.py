import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from src.models.evaluate_models import evaluate_model, save_results

def run_logistic_regression(data_path = "data/features/eng1_data_combined.csv"):
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
    y_train = train_df["ResultEncoded"] # Results used for training
    X_test = test_df[features] # Features used for testing
    y_test = test_df["ResultEncoded"] # Results used for testing

    scaler = StandardScaler() # Instance of StandardScaler is created
    X_train_scaled = scaler.fit_transform(X_train) # Training data mean and standard deviation of each feature are calculated using fit and they are then scaled using transform to have a mean if 0 and a standard deviation of 1
    X_test_scaled = scaler.transform(X_test) # Test data is scaled using transform but mean and standard deviation are not calculated since the model must not learn anything from the test data

    valid_combinations = { # Valid combinations for solvers and penalty types are defined
        "lbfgs": ["l2", None], # lbfgs can only work with l2 or no penalty
        "newton-cg": ["l2", None], # newton-cg can only work with l2 or no penalty
        "liblinear": ["l1", "l2"], # liblinear can only work with l1 and l2
        "saga": ["l1", "l2", "elasticnet", None] # saga works with l1, l2, elasticnet and no penalty
    }

    param_grid = [] # Empty list that will hold one parameter grid for each solver type is initialised

    for solver, penalties in valid_combinations.items(): # Each solver and penalty combination is looped through
        grid = { # A parameter grid is built for each combination
            "solver": [solver], # The optimization algorithm which is used
            "penalty": penalties, # Regularisation type
            "C": [0.001, 0.01, 0.1, 1, 10, 100], # Inverse of regularisation strength
            "fit_intercept": [True, False], # Specifies if an intercept term should be included
            "class_weight": [None, "balanced"], # Controls importance of each class during training
            "tol": [1e-6, 1e-4, 1e-2], # Convergence tolerance for stopping criteria
            "max_iter": [500, 1000, 2000], # Maximum number of possible iterations for solver to converge
            "warm_start": [True, False], # Whether to make use of the previous solution as a starting point
        }

        if solver == "saga" and "elasticnet" in penalties: # If saga solver and elasticnet penalty are being used l1_ratio is added
            grid["l1_ratio"] = [0.0, 0.25, 0.5, 0.75, 1.0] # Sets a combination between l1 and l2 where 0.0 = l2 only and 1.0 = l1 only

        param_grid.append(grid) # The current grid is added to the list of parameter grids

    base_model = LogisticRegression(random_state = 0) # A logistic regression model is initalised with a fixed random_state

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
    grid_search.fit(X_train_scaled, y_train) # Parameter tuning is performed on scaled data

    print("Grid Search Complete. Best Parameters:") # Prints message confirmation message that grid search completed successfully
    print(grid_search.best_params_) # Best parameters found from grid search are printed
    print(f"Best Cross Validation Accuracy: {grid_search.best_score_:.4f}") # Best cross validation accuracy is printed

    model = grid_search.best_estimator_ # Model trained using the best parameters is retrieved
    preds = model.predict(X_test_scaled) # Predicted outcomes are stored

    results = evaluate_model("Logistic Regression", y_test, preds) # Model performance is evaluated and stored in results
    save_results(results) # Model results are saved

    out = test_df.copy() # It stored a copy of the dataframe used for testing in out
    out["Predicted"] = preds # Add a predicted column to the dataframe
    out.to_csv("data/results/logistic_regression_2023_predictions.csv", index = False) # Converts the final dataframe to csv and appends it to existing csv or stores it in new csv
    print("Predictions saved to: data/results/logistic_regression_2023_predictions.csv") # Prints confirmation that the results have been stored
    return model, results # Returns the trained model and the respective results

if __name__ == "__main__":
    run_logistic_regression()