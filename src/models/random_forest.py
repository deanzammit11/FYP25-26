import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Continuous, Integer
from src.models.evaluate_models import evaluate_model, save_results

def run_random_forest(data_path = "data/features/eng1_data_combined.csv"):
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
        "OddsDifference_AvD"
    ]

    X_train = train_df[features] # Features used for training
    y_train = train_df["ResultEncoded"] # Results used for training
    X_test = test_df[features] # Features used for testing
    y_test = test_df["ResultEncoded"] # Results used for testing

    base_model = RandomForestClassifier(random_state = 0) # A random forest model is initialised with a fixed random_state

    bootstrap_spaces = { # Bootstrap specific search spaces are defined with random seed applied to space parameter to ensure consistency
        "True": { # max_samples can be tuned
            "bootstrap": Categorical([True], random_state=random_seed),
            "max_samples": Continuous(0.5, 1.0, random_state=random_seed)
        },
        "False": { # max_samples cannot be tuned
            "bootstrap": Categorical([False], random_state=random_seed)
        }
    }

    common_params = { # Common bootstrap parameters with random seed applied to each parameter to ensure consistency
        "n_estimators": Integer(100, 500, random_state=random_seed), # Number of trees in forest
        "max_depth": Integer(3, 20, random_state=random_seed), # Maximum depth of each tree
        "min_samples_split": Integer(2, 10, random_state=random_seed), # Minimum number of samples required to split an internal node
        "min_samples_leaf": Integer(1, 5, random_state=random_seed), # Minimum number of samples required to be at a leaf node
        "max_features": Categorical(["sqrt", "log2", None], random_state=random_seed), # Number of features to consider when looking for best split
        "criterion": Categorical(["gini", "entropy", "log_loss"], random_state=random_seed), # Function to measure the quality of a split
        "class_weight": Categorical([None, "balanced"], random_state=random_seed) # Controls importance of each class during training
    }

    cv = StratifiedKFold(n_splits=3, shuffle=False) # Cross validation is set to 3 fold with each fold maintaining the same ratio of outcomes as the full dataset.

    best_model = None # Variable to store the best performing model
    best_score = 0 # Variable to store the best achieved score
    best_params = None # Variable to store parameters of the best model

    for bootstrap_setting, space in bootstrap_spaces.items(): # Genetic algorithm is run for each bootstrap not to have compatibility issues
        print(f"Running Genetic Algorithm for Bootstrap: {bootstrap_setting}") # Confrimation message showing that the genetic algorithm is being run for that bootstrap setting

        base_model = RandomForestClassifier(random_state = 0) # A random forest model is initialised with a fixed random_state
        
        if bootstrap_setting == "False": # If bootstrap is False max_samples is set to None
            base_model.set_params(max_samples=None)

        param_grid = {**space, **common_params} # Bootstrap specific and common parameter spaces are combined into a single dictionary

        ga_search = GASearchCV( # Genetic Algorithm Search is setup to find the best combination of parameters
            estimator = base_model, # The model which is being optimised
            param_grid = param_grid, # The parameter combinations to test
            scoring = make_scorer(accuracy_score), # Custom scoring function is used to compute fitness based on model accuracy
            cv = cv, # Previously defined StratifiedKFold cross validator
            population_size = 25, # Number of individuals in each generation
            generations = 15, # Number of generations that the algorithm will evolve through
            n_jobs = -1, # All CPU cores are utilised
            verbose = True, # Progress is displayed in terminal
            keep_top_k = 4, # 4 best performing individuals from each generation are kept
            crossover_probability = 0.8, # Probability that two parent individuals will exchange parameter values
            mutation_probability = 0.1, # Probability that a parameter in an individual will mutate
            tournament_size = 3, # Number of individuals competing in each tournament selection event
            criteria = "max" # Scoring function will be maximised
        )

        ga_search.fit(X_train, y_train) # Parameter tuning is performed on unscaled data since Random Forest does not require feature scaling

        print(f"Bootstrap {bootstrap_setting} Complete.") # Prints confirmation message that genetic algorithm ran successfully

        if ga_search.best_score_ > best_score: # Updates best model if current solver achieves a higher accuracy
            best_score = ga_search.best_score_ # Best accuracy is overwritten
            best_model = ga_search.best_estimator_ # Best model is overwritten
            best_params = ga_search.best_params_ # Best parameters are overwritten

    print("Genetic Algorithm Complete. Best Parameters:")
    print(best_params)
    print(f"Best Cross Validation Accuracy: {best_score:.4f}")

    preds = best_model.predict(X_test) # Predicted outcomes are stored

    results = evaluate_model("Random Forest", y_test, preds) # Model performance is evaluated and stored in results
    save_results(results) # Model results are saved

    out = test_df.copy() # It stores a copy of the dataframe used for testing in out
    out["Predicted"] = preds # Adds a predicted column to the dataframe
    out.to_csv("data/results/random_forest_2023_predictions.csv", index = False) # Converts the final dataframe to csv and saves it
    print("Predictions saved to: data/results/random_forest_2023_predictions.csv") # Prints confirmation that the results have been stored

    return best_model, results # Returns the trained model and the respective results

if __name__ == "__main__":
    run_random_forest()
