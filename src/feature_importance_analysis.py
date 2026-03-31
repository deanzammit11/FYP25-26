from pathlib import Path
from joblib import load
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from src.utils import ensure_dirs

def feature_importance_analysis():
    data_path = Path("data/features/eng1_data_combined.csv") # The features dataset path is stored
    output_dir = Path("data/results/feature_importance") # The feature importance output directory path is stored
    model_config = { # A dictionary containing the selected features path, model path and the label for each model is defined
        "logistic_regression": { # A dictionary containing the logistic regression model configuration is defined
            "selected_features_path": Path("data/results/logistic regression/logistic_regression_selected_features.csv"), # The selected features csv file path is stored
            "model_path": Path("data/results/logistic regression/logistic_regression_best_model.joblib"), # The saved fitted model path is stored
            "label": "Logistic Regression", # The label is stored
        },
        "random_forest": { # A dictionary containing the random forest model configuration is defined
            "selected_features_path": Path("data/results/random forest/random_forest_selected_features.csv"), # The selected features csv file path is stored
            "model_path": Path("data/results/random forest/random_forest_best_model.joblib"), # The saved fitted model path is stored
            "label": "Random Forest", # The label is stored
        },
        "xgboost": { # A dictionary containing the xgboost model configuration is defined
            "selected_features_path": Path("data/results/xgboost/xgboost_selected_features.csv"), # The selected features csv file path is stored
            "model_path": Path("data/results/xgboost/xgboost_best_model.joblib"), # The saved fitted model path is stored
            "label": "XGBoost", # The label is stored
        },
    }

    if not data_path.exists(): # If the features dataset does not exist
        raise FileNotFoundError(f"Feature dataset not found: {data_path}") # A file not found error is raised

    ensure_dirs(str(output_dir)) # Checks if the feature importance output directory exists and if it does not it creates it
    df = pd.read_csv(data_path) # The features dataset is stored in a dataframe
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce") # The Season column is converted to numeric with any invalid values becoming null
    df["ResultEncoded"] = pd.to_numeric(df["ResultEncoded"], errors="coerce") # The ResultEncoded column is converted to numeric with any invalid values becoming null
    df = df.dropna(subset=["Season", "ResultEncoded"]).copy() # Rows missing either Season or ResultEncoded are removed and the filtered dataframe is copied
    df["Season"] = df["Season"].astype(int) # The Season column is converted to integer
    df["ResultEncoded"] = df["ResultEncoded"].astype(int) # The ResultEncoded column is converted to integer

    train_df = df[df["Season"] < 2023].copy() # Splits training data using 2019 - 2022 Seasons and stores it in a dataframe
    test_df = df[df["Season"] == 2023].copy() # Splits testing data using 2023 Season only and stores it in a dataframe
    if train_df.empty or test_df.empty: # If either the training dataframe or testing dataframe is empty
        raise ValueError("Expected non-empty train set (Season < 2023) and test set (Season == 2023).") # A value error is raised

    combined_comparison_rows = [] # An empty list which will store multiple dataframes containing the importance value for each feature for each importance method per model is initialised
    top_feature_rows = [] # An empty list which will store a dictionary for each of the top 10 features for each model and method containing the related information, importance rank and value is initialised

    for model_key, config in model_config.items(): # For each model identifier and the respective model configuration in the model configuration dictionary
        model_label = config["label"] # The label for the current model is stored
        selected_features_path = config["selected_features_path"] # The selected features csv file path for the current model is stored
        model_path = config["model_path"] # The saved fitted model path for the current model is stored
        if not selected_features_path.exists(): # If the selected features csv file for the current model does not exist
            raise FileNotFoundError(f"Selected features file not found: {selected_features_path}") # A file not found error is raised
        if not model_path.exists(): # If the saved fitted model file for the current model does not exist
            raise FileNotFoundError(f"Saved model file not found: {model_path}. The model must first be trained.") # A file not found error is raised

        selected_features_df = pd.read_csv(selected_features_path) # The selected features csv for the current model is stored in a dataframe
        if "Feature" not in selected_features_df.columns: # If the selected features dataframe does not contain a Feature column
            raise ValueError(f"Selected-features file is missing 'Feature' column: {selected_features_path}") # A value error is raised

        selected_features = selected_features_df["Feature"].dropna().astype(str).tolist() # Any missing values are removed from the Feature column and they are then converted to a string and in a list
        if not selected_features: # If the selected features list is empty
            raise ValueError(f"No features found in selected features file: {selected_features_path}") # A value error is raised

        missing_features = sorted(set(selected_features).difference(df.columns)) # The selected feature list is converted to a set and the set difference with the features dataset is computed to identify any columns which do not exist in the features dataset and they are then sorted in alphabetical order
        if missing_features: # If one or more selected features are missing from the feature dataset
            raise ValueError(f"{model_label} selected features missing from dataset: {missing_features}") # A value error is raised

        X_train = train_df[selected_features].copy() # The selected features columns are copied from the training set and stored in a dataframe
        y_train = train_df["ResultEncoded"].copy() # The target column is copied from the training set and stored in a series
        X_test = test_df[selected_features].copy() # The selected features columns are copied from the testing set and stored in a dataframe
        y_test = test_df["ResultEncoded"].copy() # The target column is copied from the testing set and stored in a series
        model = load(model_path) # The saved fitted model for the current model is loaded

        if model_key == "logistic_regression": # If the current model being analysed is logistic regression
            coefficients = model.named_steps["model"].coef_ # The fitted logistic regression coefficient matrix containing the learnt feature weights for each class is extracted
            absolute_coefficients = np.abs(coefficients) # The absolute value of each coefficient is computed so influence strength can be compared regardless of the direction
            mean_absolute = absolute_coefficients.mean(axis=0) # The mean absolute coefficient across classes for each feature column is computed
            max_absolute = absolute_coefficients.max(axis=0) # The maximum absolute coefficient across classes for each feature column is computed
            method_specific_df = pd.DataFrame( # A dataframe storing the coefficient importance values for each specific feature for logistic regression is built
                { # The coefficient importance values for each specific feature are defined in a dictionary
                    "Feature": selected_features, # The selected feature name is stored
                    "MeanAbsoluteCoefficient": mean_absolute, # The mean absolute coefficient for the respective feature is stored
                    "MaxAbsoluteCoefficient": max_absolute, # The maximum absolute coefficient for the respective feature is stored
                }
            ).sort_values(["MeanAbsoluteCoefficient", "Feature"], ascending=[False, True]).reset_index(drop=True) # The coefficient summary dataframe is sorted by mean absolute coefficient in descending order and then by feature name in alphabetical order and the row index is reset
            method_name = "Standardized Coefficients" # The model specific method label is stored
            method_value_column = "MeanAbsoluteCoefficient" # The model specific column name to be treated as the importance value is stored
            method_specific_df.to_csv(output_dir / "logistic_regression_standardized_coefficients.csv", index=False) # The logistic regression coefficient summary csv is saved to the specified directory
            plot_title = "Logistic Regression: Standardized Coefficients" # The model specific plot title is stored
            plot_path = output_dir / "logistic_regression_standardized_coefficients.png" # The model specific plot output path is stored
        elif model_key == "random_forest": # If the current model being analysed is random forest
            background = X_train # The SHAP background dataset to be used as a baseline for comparison to see how much each feature moved the prediction is set to all training rows
            explain_data = X_test # The rows which require feature explanation with SHAP are set to all testing rows
            explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional") # A SHAP TreeExplainer is created for the fitted model using the training data as the baseline for comparison 
            shap_values = explainer.shap_values(explain_data) # The SHAP values for the rows which require feature explanation are computed
            stacked = np.asarray(shap_values) # The SHAP values are converted into a numpy array with rows, features and the number of different class outputs as the dimensions
            mean_absolute_shap = np.abs(stacked).mean(axis=(0, 2)) # The mean absolute SHAP value across classes is computed so influence strength for each feature can be compared regardless of the direction and values are stored in an array
            method_specific_df = pd.DataFrame( # A dataframe storing the SHAP importance values for each specific feature for random forest is built
                { # The SHAP importance values for each specific feature are defined in a dictionary
                    "Feature": X_test.columns, # The selected feature name is stored
                    "MeanAbsoluteSHAP": mean_absolute_shap, # The mean absolute SHAP value for the respective feature is stored
                }
            ).sort_values(["MeanAbsoluteSHAP", "Feature"], ascending=[False, True]).reset_index(drop=True) # The SHAP value summary dataframe is sorted by mean absolute coefficient in descending order and then by feature name in alphabetical order and the row index is reset
            method_name = "SHAP" # The model specific method label is stored
            method_value_column = "MeanAbsoluteSHAP" # The model specific column name to be treated as the importance value is stored
            method_specific_df.to_csv(output_dir / "random_forest_shap_importance.csv", index=False) # The random forest SHAP summary is saved to csv without the row index
            plot_title = "Random Forest: Mean Absolute SHAP Values" # The model specific plot title is stored
            plot_path = output_dir / "random_forest_shap_importance.png" # The model specific plot output path is stored
        else: # If the current model being analysed is XGBoost
            background = X_train # The SHAP background dataset to be used as a baseline for comparison to see how much each feature moved the prediction is set to all training rows
            explain_data = X_test # The rows which require feature explanation with SHAP are set to all testing rows
            explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional") # A SHAP TreeExplainer is created for the fitted model using the training data as the baseline for comparison 
            shap_values = explainer.shap_values(explain_data) # The SHAP values for the rows which require feature explanation are computed
            stacked = np.asarray(shap_values) # The SHAP values are converted into a numpy array with rows, features and the number of different class outputs as the dimensions
            mean_absolute_shap = np.abs(stacked).mean(axis=(0, 2)) # The mean absolute SHAP value across classes is computed so influence strength for each feature can be compared regardless of the direction and values are stored in an array
            method_specific_df = pd.DataFrame( # A dataframe storing the SHAP importance values for each specific feature for XGBoost is built
                { # The SHAP importance values for each specific feature are defined in a dictionary
                    "Feature": X_test.columns, # The selected feature name is stored
                    "MeanAbsoluteSHAP": mean_absolute_shap, # The mean absolute SHAP value for the respective feature is stored
                }
            ).sort_values(["MeanAbsoluteSHAP", "Feature"], ascending=[False, True]).reset_index(drop=True) # The SHAP value summary dataframe is sorted by mean absolute coefficient in descending order and then by feature name in alphabetical order and the row index is reset
            method_name = "SHAP" # The model specific method label is stored
            method_value_column = "MeanAbsoluteSHAP" # The model specific column name to be treated as the importance value is stored
            method_specific_df.to_csv(output_dir / "xgboost_shap_importance.csv", index=False) # The XGBoost SHAP summary is saved to csv without the row index
            plot_title = "XGBoost: Mean Absolute SHAP Values" # The model specific plot title is stored
            plot_path = output_dir / "xgboost_shap_importance.png" # The model specific plot output path is stored

        plot_df = method_specific_df.sort_values(method_value_column, ascending=False).head(15).iloc[::-1] # The method specific dataframe containing the importance value for each feature for the respective method is sorted in descending order and the top 15 most important features are selected and the order is reversed so the most important feature appears first
        fig_height = max(6, 15 * 0.38) # The plot height is set by selecting the largest height between 6 and 15 multiplied by 0.38
        fig, ax = plt.subplots(figsize=(11, fig_height)) # A matplotlib figure and axis are created for the current model specific feature importance plot
        ax.barh(plot_df["Feature"], plot_df[method_value_column], color="#4c78a8") # A blue horizontal bar chart for the current model specific feature importance plot with the feature names on the y-axis, the importance values on the x-axis is built
        ax.set_title(plot_title) # The title for the current model specific plot is set
        ax.set_xlabel(method_value_column) # The x-axis label is set to the current model specific value column name
        ax.set_ylabel("Feature") # The y-axis label is set to Feature
        ax.grid(axis="x", alpha=0.3) # Grid lines of transparency 0.3 are added across the y-axis
        fig.tight_layout() # Layout is adjusted
        fig.savefig(plot_path, dpi=200, bbox_inches="tight") # The plot is saved to the specified path setting the resolution and cropping around the table removing extra whitespace
        plt.close(fig) # Figure is closed

        def permutation_macro_f1(estimator, X, y_true): # A function which will be used as the custom scoring function for permutation importance is defined
            predictions = np.asarray(estimator.predict(X)) # Predictions are generated by the fitted model using the provided feature matrix and the results are stored in a numpy array
            if predictions.min() >= 0 and predictions.max() <= 2 and set(np.unique(y_true)).issubset({-1, 0, 1}): # If the smallest prediction label is greater than or equal to 0 and the largest prediction label is less than or equal to 2 and the distinct values in the actual results stored in a set are a subset of the set -1, 0 and 1
                predictions = predictions - 1 # The predictions are converted back to the normal encoded labels
            return f1_score(y_true, predictions, average="macro") # Macro F1 where the F1 for each class is calculated and they are then averaged is returned

        permutation_result = permutation_importance( # Permutation importance measuring how much model performance drops when one feature is shuffled is computed for the current fitted model using the test set
            model, # The current fitted model is provided to the permutation importance routine
            X_test, # The feature matrix is provided
            y_test, # The actual target values in the testing set are provided
            scoring = permutation_macro_f1, # The custom scoring function based on macro F1 is provided for permutation importance
            n_repeats = 20, # Each feature is permuted 20 times to estimate importance more robustly
            random_state = 0, # Random state is set to 0 for reproducible results
            n_jobs = -1 # All CPU cores are utilised
        )
        permutation_df = pd.DataFrame( # A dataframe storing the permutation importance values for the current model is created
            { # The permutation importance values for each specific feature are defined in a dictionary
                "Feature": X_test.columns, # The selected feature name is stored
                "ImportanceMean": permutation_result.importances_mean, # The mean permutation importance value for the respective feature is stored
                "ImportanceStd": permutation_result.importances_std, # The standard deviation of the permutation importance value for the respective feature is stored
            }
        ).sort_values(["ImportanceMean", "Feature"], ascending=[False, True]).reset_index(drop=True) # The permutation importance value summary dataframe is sorted by mean importance in descending order and then by feature name in alphabetical order and the row index is reset
        permutation_df.to_csv(output_dir / f"{model_key}_permutation_importance.csv", index=False) # The permutation importance value summary csv is saved to the specified directory

        permutation_plot_df = permutation_df.sort_values("ImportanceMean", ascending=False).head(15).iloc[::-1] # The permutation importance dataframe containing the importance value for each feature is sorted in descending order and the top 15 most important features are selected and the order is reversed so the most important feature appears first
        fig_height = max(6, 15 * 0.38) # The plot height is set by selecting the largest height between 6 and 15 multiplied by 0.38
        fig, ax = plt.subplots(figsize=(11, fig_height)) # A matplotlib figure and axis are created for the permutation importance plot
        ax.barh(permutation_plot_df["Feature"], permutation_plot_df["ImportanceMean"], color="#4c78a8") # A blue horizontal bar chart for the permutation importance plot with the feature names on the y-axis, the importance values on the x-axis is built
        ax.set_title(f"{model_label}: Permutation Importance") # The title for the perumation importance plot is set
        ax.set_xlabel("ImportanceMean") # The x-axis label is set
        ax.set_ylabel("Feature") # The y-axis label is set
        ax.grid(axis="x", alpha=0.3) # Grid lines of transparency 0.3 are added across the y-axis
        fig.tight_layout() # Layout is adjusted
        fig.savefig(output_dir / f"{model_key}_permutation_importance.png", dpi=200, bbox_inches="tight") # The plot is saved to the specified path setting the resolution and cropping around the table removing extra whitespace
        plt.close(fig) # Figure is closed

        permutation_comparison_df = permutation_df[["Feature", "ImportanceMean"]].copy() # The feature and importance mean columns are copied from the permutation importance dataframe and are stored in a new dataframe
        permutation_comparison_df.insert(0, "Method", "Permutation Importance") # A Method column with the value Permutation Importance is added as the first column of the permutation comparison dataframe
        permutation_comparison_df.insert(0, "Model", model_label) # A Model column with the current model label is added as the first column of the permutation comparison dataframe
        permutation_comparison_df = permutation_comparison_df.rename(columns={"ImportanceMean": "ImportanceValue"}) # The importance mean column is renamed to ImportanceValue
        combined_comparison_rows.append(permutation_comparison_df) # The permutation comparison dataframe for the current model is appended to the combined comparison rows list

        method_comparison_df = method_specific_df[["Feature", method_value_column]].copy() # The feature and model specific importance columns are copied from the current model specific dataframe and are stored in a new dataframe
        method_comparison_df.insert(0, "Method", method_name) # A Method column with the model specific method label is added as the first column of the model specific comparison dataframe
        method_comparison_df.insert(0, "Model", model_label) # A Model column with the current model label is inserted as the first column of the model specific comparison dataframe
        method_comparison_df = method_comparison_df.rename(columns={method_value_column: "ImportanceValue"}) # The model specific importance column is renamed to ImportanceValue
        combined_comparison_rows.append(method_comparison_df) # The model specific comparison dataframe for the current model is appended to the combined comparison rows list

        for rank, (_, row) in enumerate(permutation_df.head(10).iterrows(), start=1): # For each row of the top 10 rows in the current model permutation importance dataframe with a counter starting from 1 representing the feature rank
            top_feature_rows.append( # The row feature information for permutation importance is appended to the top feature rows list
                { # The row feature information for permuation importance is defined in a dictionary
                    "Model": model_label, # The current model label is stored
                    "Method": "Permutation Importance", # The current model method label is stored
                    "Rank": rank, # The current rank number is stored
                    "Feature": row["Feature"], # The feature name for the current row is stored
                    "ImportanceValue": row["ImportanceMean"] # The model specific permutation importance value for the current row is stored
                }
            )
        for rank, (_, row) in enumerate(method_specific_df.head(10).iterrows(), start=1): # For each row of the top 10 rows in the current model specific method importance dataframe with a counter starting from 1 representing the feature rank
            top_feature_rows.append( # The row feature information for the model specific method is appended to the top feature rows list
                { # The row feature information for the model specific method is defined in a dictionary
                    "Model": model_label, # The current model label is stored
                    "Method": method_name, # The current model specific method label is stored
                    "Rank": rank, # The current rank number is stored
                    "Feature": row["Feature"], # The feature name for the current row is stored
                    "ImportanceValue": row[method_value_column] # The model specific method importance value for the current row is stored
                }
            )

    combined_comparison_df = pd.concat(combined_comparison_rows, ignore_index=True) # All the model comparison dataframes are concatenated with the row index being reset
    combined_comparison_df.to_csv(output_dir / "feature_importance_method_comparison.csv", index=False) # The model comparison dataframe csv is saved to the specified directory

    top_features_df = pd.DataFrame(top_feature_rows) # The top 10 feature rows for all models and methods list is converted into a dataframe
    top_features_df.to_csv(output_dir / "feature_importance_top10_by_model_and_method.csv", index=False) # The top 10 features dataframe csv is saved to the specified directory

    overlap_rows = [] # An empty list which will store how much the top 10 permutation importance and top 10 model specific method features overlap between each model is initialised
    for model_label in top_features_df["Model"].drop_duplicates(): # For each model label in the top features dataframe after dropping duplicates
        model_top = top_features_df[top_features_df["Model"] == model_label] # The top features dataframe is filtered for rows of the current model only
        permutation_top = set(model_top[model_top["Method"] == "Permutation Importance"]["Feature"].head(10).tolist()) # The filtered top features dataframe is filtered for rows where the method is Permutation Importance and the Feature names for the first 10 rows are stored in a list and converted into a set
        alternate_method = next(method for method in model_top["Method"].drop_duplicates() if method != "Permutation Importance") # For each unique method name in the filtered top features dataframe select the first method which is not Permutation Importance
        alternate_top = set(model_top[model_top["Method"] == alternate_method]["Feature"].head(10).tolist()) # The filtered top features dataframe is filtered for rows where the method is the model specific method and the Feature names for the first 10 rows are stored in a list and converted into a set
        overlap_rows.append( # The information for the overlap between the top 10 rows is appended to the overlap rows list
            { # The information for the overlap between the top 10 rows is defined in a dictionary
                "Model": model_label, # The current model label is stored
                "MethodA": "Permutation Importance", # The first method label is stored as Permutation Importance
                "MethodB": alternate_method, # The second method label is stored as the current model specific method
                "Top10OverlapCount": len(permutation_top.intersection(alternate_top)), # The set intersection between the top 10 permuation importance features and the top 10 model specific method features is computed and the number of elements in the set is stored
                "SharedFeatures": ", ".join(sorted(permutation_top.intersection(alternate_top))) # The set intersection between the top 10 permuation importance features and the top 10 model specific method features is computed and they are the features are then sorted in alphabetical order and joined using a comma
            }
        )
    pd.DataFrame(overlap_rows).to_csv(output_dir / "feature_importance_top10_overlap.csv", index=False) # The overlap summary rows list is converted into a dataframe and the csv is saved to the specified directory

    print(f"Feature-importance outputs saved under: {output_dir}") # A confirmation message showing where the feature importance outputs were saved is printed

if __name__ == "__main__":
    feature_importance_analysis()