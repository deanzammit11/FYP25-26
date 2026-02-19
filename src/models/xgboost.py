import numpy as np
import pandas as pd
import random
import os
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.space import Categorical, Continuous, Integer
from src.models.evaluate_models import evaluate_model, save_results
from src.models.sequential_elo import predict_2023_with_elo_updates

def run_xgboost(data_path = "data/features/eng1_data_combined.csv"):
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
    y_train = train_df["ResultEncoded"] + 1 # Results used for training and adding 1 since xgboost requires match outcomes to be in the form of 0, 1 and 2 not -1, 0 and 1
    X_test = test_df[features] # Features used for testing
    y_test = test_df["ResultEncoded"] + 1 # Results used for testing and adding 1 since xgboost requires match outcomes to be in the form of 0, 1 and 2 not -1, 0 and 1
    use_feature_selection = True # Boolean variable which determines if feature selection is to be used
    
    base_model = XGBClassifier(
        objective = "multi:softprob", # Multi-class classification outputting class probabilities.
        num_class = 3, # Refers to 3 possible outcome classes
        eval_metric = "mlogloss", # Measures how well predicted probabilities match true labels.
        random_state = 0, # Ensure results are reproducible and consistent
        n_jobs = -1, # Enables usage of all available CPU cores
        tree_method = "hist", # Uses a histogram based algorithm for building the trees
    )    

    param_grid = { # Random seed applied to each parameter to ensure consistency
        "learning_rate": Continuous(0.01, 0.3, random_state=random_seed), # Step size at each boosting step
        "n_estimators": Integer(300, 500, random_state=random_seed), # Number of trees to build
        "max_depth": Integer(4, 8, random_state=random_seed), # Maximum depth of each tree
        "min_child_weight": Integer(1, 5, random_state=random_seed), # Minimum sum of instance weights in a child node
        "gamma": Continuous(0, 2, random_state=random_seed), # Minimum loss reduction required to further split a leaf node
        "subsample": Continuous(0.6, 1.0, random_state=random_seed), # Fraction of training samples used for each boosting round
        "colsample_bytree": Continuous(0.6, 1.0, random_state=random_seed), # Fraction of features used by tree
        "reg_lambda": Continuous(0.1, 3.0, random_state=random_seed), # L2 regularisation term
        "reg_alpha": Continuous(0.0, 1.0, random_state=random_seed), # L1 regularisation term
        "scale_pos_weight": Continuous(1.0, 1.5, random_state=random_seed), # Adjusts for class imbalance where 1 = no correction and 1.5 = slightly favors minority classes
        "grow_policy": Categorical(["depthwise"], random_state=random_seed) # Tree growth strategy
    }

    groups = train_df["Season"].to_numpy() # The season for each row in the training set is stored
    cv = StratifiedGroupKFold(n_splits=4, shuffle=False) # Cross validation is set to 4 folds with each fold consisting of a whole season
    cv_splits = list(cv.split(X_train, y_train, groups=groups)) # Splits the training set into equal folds and grouping by season

    os.makedirs("data/results/xgboost", exist_ok=True) # Checks if directory for output file exists and if not it creates it
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
    }).to_csv("data/results/xgboost/xgboost_cv_folds.csv", index=False) # Converts it to csv and appends it to existing csv or stores it in new csv
    print("Folds saved to: data/results/xgboost/xgboost_cv_folds.csv") # Prints confirmation that the folds have been stored

    X_train_selected = X_train # The default setting is all the features used for training
    X_test_selected = X_test # The default setting is all the features used for testing
    selected_features = features # The default setting is all the features in the features list
    if use_feature_selection: # If feature selection will be used
        print("Running Genetic Algorithm Feature Selection") # Print confirmation message that feature selection has started

        feature_selection_estimator = XGBClassifier( # The model which will be used as an estimator to select features is defined
            objective = "multi:softprob", # multi:softprob is selected as the objectivr for the basic model
            num_class = 3, # The number of classes is set to 3
            eval_metric = "mlogloss", # mlogloss is used as the evaluation metric
            random_state = 0, # Random state is set to 0 for reproducible results
            n_jobs = -1, # n_jobs is set to -1 so that all CPU cores can be used
            tree_method = "hist", # The histogram based tree builder is used
        )

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

        pd.DataFrame({"Feature": selected_features}).to_csv("data/results/xgboost/xgboost_selected_features.csv", index=False) # The selected features are stored in a csv file in the specified directory
        print("Selected features saved to: data/results/xgboost/xgboost_selected_features.csv") # Prints confirmation that the selected features have been stored

    ga_search = GASearchCV( # Grid search is setup to find the best combination of parameters through cross validation
        estimator = base_model, # The model which is being optmised
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
        tournament_size = 3, # # Number of individuals competing in each tournament selection event
        criteria = "max" # Scoring function will be maximised
    )

    print("Running Genetic Algorithm.") # Prints message to reassure that the genetic algorithm is running
    ga_search.fit(X_train_selected, y_train) # Parameter tuning is performed on data using precomputed season-grouped folds

    print("Genetic Algorithm Complete. Best Parameters:") # Prints confirmation message that genetic algorithm completed successfully
    print(ga_search.best_params_) # Best parameters found from the genetic algorithm are printed
    print(f"Best Cross Validation F1: {ga_search.best_score_:.3f}") # Best cross validation F1 score is printed

    model = ga_search.best_estimator_ # Model trained using the best parameters is retrieved
    preds, test_df = predict_2023_with_elo_updates(model=model, test_df=test_df, feature_columns=selected_features, prediction_to_result=lambda pred: int(pred) - 1) # Predicted outcomes are generated with Elo updated after each predicted fixture
    y_test = y_test - 1 # Subtracting 1 to convert testing outcomes back to the original format

    results = evaluate_model("XGBoost", y_test, preds) # Model performance is evaluated and stored in results
    save_results(results) # Model results are saved

    out = test_df.copy() # It stored a copy of the dataframe used for testing in out
    out["Predicted"] = preds # Add a predicted column to the dataframe
    out.to_csv("data/results/xgboost/xgboost_2023_predictions.csv", index = False) # Converts the final dataframe to csv and appends it to existing csv or stores it in new csv
    print("Predictions saved to: data/results/xgboost/xgboost_2023_predictions.csv") # Prints confirmation that the results have been stored
    return model, results # Returns the trained model and the respective results

if __name__ == "__main__":
    run_xgboost()
