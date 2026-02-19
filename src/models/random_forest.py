import numpy as np
import pandas as pd
import random
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.space import Categorical, Continuous, Integer
from src.models.evaluate_models import evaluate_model, save_results
from src.calculate_elo import predict_2023_with_elo_updates

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
        "OddsDifference_AvD",
        "HomeForm",
        "AwayForm",
        "HomeAdvantageIndex",
        "HomeGeneralForm",
        "AwayGeneralForm",
        "GeneralFormDifference",
        "AverageGoalsScoredAtHome",
        "AverageGoalsScoredAtAway",
        "AverageGoalsConcededAtHome",
        "AverageGoalsConcededAtAway",
        "TotalGoalsScoredHome",
        "TotalGoalsScoredAway",
        "TotalGoalsConcededHome",
        "TotalGoalsConcededAway",
        "WinStreakHome",
        "WinStreakAway",
        "LossStreakHome",
        "LossStreakAway",
        "TotalWinsHome",
        "TotalWinsAway",
        "TotalDrawsHome",
        "TotalDrawsAway",
        "TotalLossesHome",
        "TotalLossesAway",
        "HistoricalEncountersHome", 
        "HistoricalEncountersAway",
        "HomeFifaOverall",
        "HomeFifaAttack",
        "HomeFifaMidfield",
        "HomeFifaDefence",
        "AwayFifaOverall",
        "AwayFifaAttack",
        "AwayFifaMidfield",
        "AwayFifaDefence",
        "HFA",
        "HomeElo", 
        "AwayElo",
        "EloTierHome",
        "EloTierAway",
    ]

    X_train = train_df[features] # Features used for training
    y_train = train_df["ResultEncoded"] # Results used for training
    X_test = test_df[features] # Features used for testing
    y_test = test_df["ResultEncoded"] # Results used for testing

    use_feature_selection = True # Boolean variable which determines if feature selection is to be used

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

    groups = train_df["Season"].to_numpy() # The season for each row in the training set is stored
    cv = StratifiedGroupKFold(n_splits=4, shuffle=False) # Cross validation is set to 4 folds with each fold consisting of a whole season
    cv_splits = list(cv.split(X_train, y_train, groups=groups)) # Splits the training set into equal folds and grouping by season

    os.makedirs("data/results/random forest", exist_ok=True) # Checks if directory for output file exists and if not it creates it
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
    }).to_csv("data/results/random forest/random_forest_cv_folds.csv", index=False) # Converts it to csv and appends it to existing csv or stores it in new csv
    print("Folds saved to: data/results/random forest/random_forest_cv_folds.csv") # Prints confirmation that the folds have been stored

    X_train_selected = X_train # The default setting is all the features used for training
    X_test_selected = X_test # The default setting is all the features used for testing
    selected_features = features # The default setting is all the features in the features list
    if use_feature_selection: # If feature selection will be used
        print("Running Genetic Algorithm Feature Selection") # Print confirmation message that feature selection has started

        feature_selection_estimator = RandomForestClassifier(random_state = 0) # The model which will be used as an estimator to select features is defined

        ga_feature_select = GAFeatureSelectionCV(
            estimator = feature_selection_estimator, # The model which is being optimised
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

        ga_feature_select.fit(X_train, y_train) # Feature selection is performed on data using precomputed season-grouped folds

        support_mask = ga_feature_select.support_ # The feature boolean mask returned by feature selection is stored
        if not np.any(support_mask): # If all values in the mask are false
            print("Feature selection returned no features. Using all features instead.") # Print confirmation message that no features were returned and that all the features will be used
            support_mask = np.ones(len(features), dtype=bool) # The mask is replaced with a new one which returns true for each feature

        selected_features = [name for name, keep in zip(features, support_mask) if keep] # The selected features list is built uisng list comprehension

        X_train_selected = X_train.loc[:, selected_features] # Selects all rows and only the selected features for each row in the training set and stores them
        X_test_selected = X_test.loc[:, selected_features] # Selects all rows and only the selected features for each row in the testing set and stores them

        pd.DataFrame({"Feature": selected_features}).to_csv("data/results/random forest/random_forest_selected_features.csv", index=False) # The selected features are stored in a csv file in the specified directory
        print("Selected features saved to: data/results/random forest/random_forest_selected_features.csv") # Prints confirmation that the selected features have been stored

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
            scoring = make_scorer(f1_score, average="macro"), # Custom scoring function is used to compute fitness based on model f1 score
            cv = cv_splits, # Previously defined StratifiedKFold cross validator
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

        ga_search.fit(X_train_selected, y_train) # Parameter tuning is performed on data using precomputed season-grouped folds

        print(f"Bootstrap {bootstrap_setting} Complete.") # Prints confirmation message that genetic algorithm ran successfully

        if ga_search.best_score_ > best_score: # Updates best model if current solver achieves a higher accuracy
            best_score = ga_search.best_score_ # Best accuracy is overwritten
            best_model = ga_search.best_estimator_ # Best model is overwritten
            best_params = ga_search.best_params_ # Best parameters are overwritten

    print("Genetic Algorithm Complete. Best Parameters:")
    print(best_params)
    print(f"Best Cross Validation F1: {best_score:.3f}")

    preds, test_df = predict_2023_with_elo_updates(model=best_model, test_df=test_df, feature_columns=selected_features) # Predicted outcomes are generated with Elo updated after each predicted fixture

    results = evaluate_model("Random Forest", y_test, preds) # Model performance is evaluated and stored in results
    save_results(results) # Model results are saved

    out = test_df.copy() # It stores a copy of the dataframe used for testing in out
    out["Predicted"] = preds # Adds a predicted column to the dataframe
    out.to_csv("data/results/random forest/random_forest_2023_predictions.csv", index = False) # Converts the final dataframe to csv and saves it
    print("Predictions saved to: data/results/random forest/random_forest_2023_predictions.csv") # Prints confirmation that the results have been stored

    return best_model, results # Returns the trained model and the respective results

if __name__ == "__main__":
    run_random_forest()
