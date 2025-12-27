import numpy as np
import pandas as pd
import random
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Continuous, Integer
from src.models.evaluate_models import evaluate_model, save_results

def run_logistic_regression(data_path = "data/features/eng1_data_combined.csv"):
    random_seed = 0 # Seed value is set to 0
    np.random.seed(random_seed) # Random seed for numpy random number generator is set
    random.seed(random_seed) # Random seed for python random number generator is set

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
        "OddsDifference_AvD",
        "HomeForm",
        "AwayForm",
        "HomeAdvantageIndex",
        "HomeGeneralForm",
        "AwayGeneralForm",
        "GeneralFormDifference",
        # "AverageGoalsScoredAtHome",
        # "AverageGoalsScoredAtAway",
        # "AverageGoalsConcededAtHome",
        # "AverageGoalsConcededAtAway",
        # "TotalGoalsScoredHome",
        # "TotalGoalsScoredAway",
        # "TotalGoalsConcededHome",
        # "TotalGoalsConcededAway",
        "HistoricalEncountersHome", 
        "HistoricalEncountersAway",
        "HomeFifaOverall",
        # "HomeFifaAttack",
        # "HomeFifaMidfield",
        # "HomeFifaDefence",
        "AwayFifaOverall",
        # "AwayFifaAttack",
        # "AwayFifaMidfield",
        # "AwayFifaDefence"
    ]

    X_train = train_df[features] # Features used for training
    y_train = train_df["ResultEncoded"] # Results used for training
    X_test = test_df[features] # Features used for testing
    y_test = test_df["ResultEncoded"] # Results used for testing

    scaler = StandardScaler() # Instance of StandardScaler is created
    X_train_scaled = scaler.fit_transform(X_train) # Training data mean and standard deviation of each feature are calculated using fit and they are then scaled using transform to have a mean if 0 and a standard deviation of 1
    X_test_scaled = scaler.transform(X_test) # Test data is scaled using transform but mean and standard deviation are not calculated since the model must not learn anything from the test data

    base_model = LogisticRegression(random_state = 0, warm_start = False) # A logistic regression model is initalised with a fixed random_state

    solver_spaces = { # Solver specific search spaces are defined with random seed applied to space parameter to ensure consistency
        "lbfgs": { # lbfgs can only work with l2 or no penalty
            "solver": Categorical(["lbfgs"], random_state=random_seed),
            "penalty": Categorical(["l2", None], random_state=random_seed)
        },
        "newton-cg": { # newton-cg can only work with l2 or no penalty
            "solver": Categorical(["newton-cg"], random_state=random_seed),
            "penalty": Categorical(["l2", None], random_state=random_seed)
        },
        "liblinear": { # liblinear can only work with l1 and l2
            "solver": Categorical(["liblinear"], random_state=random_seed),
            "penalty": Categorical(["l1", "l2"], random_state=random_seed)
        },
        "saga": { # saga works with l1, l2, elasticnet and no penalty
            "solver": Categorical(["saga"], random_state=random_seed),
            "penalty": Categorical(["l1", "l2", "elasticnet", None], random_state=random_seed),
            "l1_ratio": Continuous(0.0, 1.0, random_state=random_seed)
        }
    }

    common_params = { # Common solver parameters with random seed applied to each parameter to ensure consistency
        "C": Continuous(0.001, 100.0, random_state=random_seed), # Inverse of regularisation strength
        "fit_intercept": Categorical([True, False], random_state=random_seed), # Specifies if an intercept term should be included
        "class_weight": Categorical([None, "balanced"], random_state=random_seed), # Controls importance of each class during training
        "tol": Continuous(1e-6, 1e-2, distribution="log-uniform", random_state=random_seed), # Convergence tolerance for stopping criteria
        "max_iter": Integer(500, 2000, random_state=random_seed) # Maximum number of possible iterations for solver to converge
    }

    groups = train_df["Season"].to_numpy() # The season for each row in the training set is stored
    cv = StratifiedGroupKFold(n_splits=4, shuffle=False) # Cross validation is set to 4 folds with each fold consisting of a whole season
    cv_splits = list(cv.split(X_train, y_train, groups=groups)) # Splits the training set into equal folds and grouping by season

    os.makedirs("data/results/logistic regression", exist_ok=True) # Checks if directory for output file exists and if not it creates it
    outcome_labels = train_df["ResultEncoded"] # Captures outcome labels from training set
    row_fold = np.full(len(X_train), -1, dtype=int) # Creates a numpy array of the same length as the training set with each index filled with a placeholder value of -1
    fold_order = {} # An empty dictionary to store which season each fold represents is defined
    for fold_idx, (train_idx, validation_idx) in enumerate(cv_splits, start=1): # Splits the training set into equal folds grouping by season and starts counting the folds from 1 instead of 0
        seasons_in_fold = np.unique(groups[validation_idx]) # The seasons in the respective fold are captured
        if len(seasons_in_fold) == 0: # Checks that each fold contains at least one season
            raise ValueError("Each fold should contain at least one season.") # An error is printed if there is more than one season or no season in a fold
        fold_order[fold_idx] = seasons_in_fold.min() # Earliest season for the respective fold is picked
    sorted_folds = sorted(fold_order.items(), key=lambda item: item[1]) # Folds are sorted by season
    ordered_folds = {} # An empty dictionary to store the mapping between the old fold id and the new fold id is defined
    order = 1 # Order counter is defined to 1
    for original_fold, season in sorted_folds: # For each sorted fold to season mapping
        ordered_folds[original_fold] = order # Maps the original fold to the new fold
        order += 1 # Increments order by 1

    for fold, (train_idx, validation_idx) in enumerate(cv_splits, start=1): # Splits the training set into equal folds grouping by season and starts counting the folds from 1 instead of 0
        row_fold[validation_idx] = ordered_folds[fold] # Overwrites the old fold with the new one based on the mapping in ordered_folds

    pd.DataFrame({
        "Row Number": np.arange(1, len(X_train) + 1), # Stores the row number starting from 1 up to N
        "Fold": row_fold, # The fold number is stored
        "Season": train_df["Season"].to_numpy(), # The season is stored
        "Outcome": outcome_labels.to_numpy(), # The result is stored
    }).to_csv("data/results/logistic regression/logistic_regression_cv_folds.csv", index=False) # Converts it to csv and appends it to existing csv or stores it in new csv
    print("Folds saved to: data/results/logistic regression/logistic_regression_cv_folds.csv") # Prints confirmation that the folds have been stored

    best_model = None # Variable to store the best performing model is defined
    best_score = 0 # Variable to store the best achieved score is defined
    best_params = None # Variable to store parameters of the best model is defined

    for solver_name, space in solver_spaces.items(): # Genetic algorithm is run for each solver not to have compatibility issues
        print(f"Running Genetic Algorithm for Solver: {solver_name}") # Confrimation message showing that the genetic algorithm is being run for that solver

        param_grid = {**space, **common_params} # Solver specific and common parameter spaces are combined into a single dictionary

        ga_search = GASearchCV( # Genetic Algorithm Search is setup to find the best combination of parameters
            estimator = base_model, # The model which is being optimised
            param_grid = param_grid, # The parameter combinations to test
            scoring = make_scorer(f1_score, average="macro"), # Custom scoring function is used to compute fitness based on model f1 score
            cv = cv_splits, # Previously defined StratifiedGroupKFold cross validator grouped by season
            population_size = 25, # Number of individuals in each generation
            generations = 15, # Number of generations that the algorithm will evolve through
            n_jobs = -1, # All CPU cores are utilised
            verbose = True, # Progress is displayed in terminal
            keep_top_k = 4, # 4 best performing individuals from each generation are kept
            crossover_probability = 0.8, # Probability that two parent individuals will exchange parameter values
            mutation_probability = 0.1, # Probability that a parameter in an individual will mutate
            tournament_size = 3, # Number of individuals competing in each tournament selection event
            criteria = "max", # Scoring function will be maximised
        )

        ga_search.fit(X_train_scaled, y_train) # Parameter tuning is performed on scaled data using precomputed season-grouped folds

        print(f"Solver {solver_name} Complete.") # Prints confirmation message that genetic algorithm ran successfully

        if ga_search.best_score_ > best_score: # Updates best model if current solver achieves a higher accuracy
            best_score = ga_search.best_score_ # Best accuracy is overwritten
            best_model = ga_search.best_estimator_ # Best model is overwritten
            best_params = ga_search.best_params_ # Best parameters are overwritten

    print("Genetic Algorithm Complete. Best Parameters:") # Prints confirmation message that genetic algorithm completed successfully
    print(best_params) # Best parameters found from the genetic algorithm are printed
    print(f"Best Cross Validation Accuracy: {best_score:.3f}") # Best cross validation accuracy is printed

    preds = best_model.predict(X_test_scaled) # Predicted outcomes are stored

    results = evaluate_model("Logistic Regression", y_test, preds) # Model performance is evaluated and stored in results
    save_results(results) # Model results are saved

    out = test_df.copy() # It stored a copy of the dataframe used for testing in out
    out["Predicted"] = preds # Add a predicted column to the dataframe
    out.to_csv("data/results/logistic regression/logistic_regression_2023_predictions.csv", index = False) # Converts the final dataframe to csv and appends it to existing csv or stores it in new csv
    print("Predictions saved to: data/results/logistic regression/logistic_regression_2023_predictions.csv") # Prints confirmation that the results have been stored

    return best_model, results # Returns the trained model and the respective results

if __name__ == "__main__":
    run_logistic_regression()
