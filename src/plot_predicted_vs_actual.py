import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import ensure_dirs, extract_model_name_from_filename

def plot_predicted_vs_actual_by_season(predictions_path: str = "data/results/*/*[0-9][0-9][0-9][0-9]_predictions.csv", output_path: str = "data/results"):
    ensure_dirs(output_path) # Checks if directory exists and if it does not it creates it

    paths = sorted(glob.glob(predictions_path)) # All prediction file paths which match predictions_path are stored in files
    
    if not paths: # If no paths are found
        print(f"No prediction files found for pattern: {predictions_path}") # A message stating that no prediction files were found is printed
    else:
        for file_path in paths: # For each predictions csv file
            model = extract_model_name_from_filename(file_path) # The model name is extracted from the filename
            df = pd.read_csv(file_path) # The predictions csv is loaded

            base_results_path = os.path.join("data", "results") # Creates data/results as the base path by joining data and results since all model subfolders lie in there
            try:
                path_difference = os.path.relpath(file_path, base_results_path) # Captures the difference of the predictions file path relative to data/results
                model_folder = path_difference.split(os.sep)[0] # Splits the path difference with / acting as the seperator and takes the first element which will always be the model name
                output_path = os.path.join(base_results_path, model_folder) # Joins the base path with the model folder to obtain the final path for that model
            except ValueError:
                output_path = os.path.dirname(file_path) # The path for the predictions file is used if the path difference cannot be computed
            ensure_dirs(output_path) # Checks if directory exists and if it does not it creates it

            required_columns = {"Season", "ResultEncoded", "Predicted"} # The required columns are defined
            missing = required_columns - set(df.columns) # Missing required columns are identified by working out the set difference
            
            if missing: # If there are any missing columns the file is skipped
                print(f"Skipping {file_path} because there are missing columns: {missing}") # A message stating that there are missing columns for the specific predictions file is printed along with the missing columns
            else:
                df = df.dropna(subset=["Season", "ResultEncoded", "Predicted"]).copy() # Rows whose any of the required columns are null are dropped
                df["Season"] = df["Season"].astype(int) # Season values are type casted as integer

                df["ActualOutcome"] = df["ResultEncoded"].astype(int).map({1: "Home Win", 0: "Draw", -1: "Away Win"}) # Outcomes are type casted to integer and are mapped to actual outcomes
                df["PredictedOutcome"] = df["Predicted"].astype(int).map({1: "Home Win", 0: "Draw", -1: "Away Win"}) # Predictions are type casted to integer and are mapped to predicted outcomes

                incorrect_flag = df["Predicted"].astype(int) != df["ResultEncoded"].astype(int) # The predicted outcomes and the actual outcome are compared and if they match true is returned else false is returned and they are stored in a boolean array
                incorrect = df.loc[incorrect_flag].copy() # The incorrectly predicted rows are stored in incorrect
                output_path_incorrect = os.path.join(output_path, f"{model}_incorrect_predictions.csv") # Output path for incorrect predictions is defined
                incorrect.sort_values(["Season"]).to_csv(output_path_incorrect, index=False)  # Incorrect predictions are sorted by season and are saved to a csv file

                actual_count = (
                    df.groupby(["Season", "ActualOutcome"]) # Groups of Season and ActualOutcome are formed
                    .size() # Number of rows are counted
                    .rename("Count") # The group is renamed to Count
                    .reset_index() # The grouped index index is converted back into normal columns
                )
                actual_count["Source"] = "Actual" # Values for Source column are set to Actual
                actual_count = actual_count.rename(columns={"ActualOutcome": "Outcome"}) # ActualOutcome column is renamed to Outcome

                predicted_count = (
                    df.groupby(["Season", "PredictedOutcome"]) # Groups of Season and ActualOutcome are formed
                    .size() # Number of rows are counted
                    .rename("Count") # The group is renamed to Count
                    .reset_index() # The grouped index index is converted back into normal columns
                )
                predicted_count["Source"] = "Predicted" # Values for Source column are set to Predicted
                predicted_count = predicted_count.rename(columns={"PredictedOutcome": "Outcome"}) # PredictedOutcome column is renamed to Outcome

                combined_count = pd.concat([actual_count, predicted_count], ignore_index=True)  # Actual and Predicted counts are combined

                table = (
                    combined_count.pivot_table( # Counts are converted into a table
                        index="Season", # A row for each season is created
                        columns=["Outcome", "Source"], # Multindex for the columns is created with the first level being the outcomes and the second level being the source
                        values="Count", # Cell values are the Count values
                        fill_value=0, # If there is a pair without a value that pair is assigned the value 0
                        aggfunc="sum", # If there is a pair which for some reason is repeated that pair is summed
                    )
                    .reindex( # Table is outputted in the form Outcome, Source
                        columns=pd.MultiIndex.from_product([["Home Win", "Draw", "Away Win"], ["Actual", "Predicted"]]), # Columns are the cartesian product
                        fill_value=0, # If there is a pair without a value that pair is assigned the value 0
                    )
                    .sort_index() # Rows are sorted by season in ascending order
                )

                ax = table.plot(kind="bar", figsize=(14, 6)) # A grouped bar chart is created using the data from table
                ax.set_title(f"{model}: Actual vs Predicted Outcomes by Season (Home/Away/Draw)") # Title for bar chart is set
                ax.set_xlabel("Season") # X-axis label is set
                ax.set_ylabel("Number of Matches") # Y-axis label is set
                ax.legend(title="Outcome, Source", bbox_to_anchor=(1.02, 1), loc="upper left") # Title for legend is set and legend is positioned outside the plot
                plt.tight_layout() # Layout is adjusted

                output_png = os.path.join(output_path, f"{model}_actual_vs_predicted_by_season.png") # Output path for the image is defined
                plt.savefig(output_png, dpi=200) # Figure is saved to the specified path and resolution is set
                plt.close() # Figure is closed

                output_csv = os.path.join(output_path, f"{model}_actual_vs_predicted_by_season.csv") # Output path for the csv is defined
                combined_count.sort_values(["Season", "Outcome", "Source"]).to_csv(output_csv, index=False) # Actual vs predicted figures are saved to csv

                cross_table = pd.crosstab( # A cross tabulation table over incorrect rows is built
                    df.loc[incorrect_flag, "PredictedOutcome"], # Rows are the incorrectly predicted values
                    df.loc[incorrect_flag, "ActualOutcome"], # Columns are the actual outcome for the incorrectly predicted rows
                ).reindex(
                    index=["Home Win", "Draw", "Away Win"], # Row order is specified
                    columns=["Home Win", "Draw", "Away Win"], # Column order is specified
                    fill_value=0 # Missing rows and columns combinations are filled with 0
                )

                pairs = [("Home Win", "Draw"), ("Home Win", "Away Win"), ("Draw", "Home Win"), ("Draw", "Away Win"), ("Away Win", "Home Win"), ("Away Win", "Draw")] # The predicted to actual pairs are defined

                mappings = [f"Pred {p} → Actual {a}" for (p, a) in pairs] # Mappings are built from pairs
                mappings_count = [int(cross_table.loc[p, a]) for (p, a) in pairs] # Each count for each possible mapping is looked up from the cross tabulation table
                mappings_df = pd.DataFrame({"Mapping": mappings, "Count": mappings_count}) # The mappings and respective count are built in a 2 column data frame

                ax2 = mappings_df.plot(kind="bar", x="Mapping", y="Count", legend=False, figsize=(14, 6)) # A bar chart is created for incorrect predictions
                ax2.set_title(f"{model}: Incorrect Predictions (Predicted → Actual)") # Title for bar chart is set
                ax2.set_xlabel("Mappings") # X-axis label is set
                ax2.set_ylabel("Number of Matches") # Y-axis label is set
                plt.xticks(rotation=30, ha="right") # X-axis labels are rotated 30 degrees anti clockwise for better readability
                plt.tight_layout() # Layout is adjusted to prevent clipping

                output_incorrect_mappings_png = os.path.join(output_path, f"{model}_incorrect_predicted_to_actual.png") # Output path for the image is defined
                plt.savefig(output_incorrect_mappings_png, dpi=200) # Figure is saved to the specified path and resolution is set
                plt.close() # Figure is closed

                incorrect_mappings_csv = os.path.join(output_path, f"{model}_incorrect_predicted_to_actual.csv") # Output path for the csv is defined
                mappings_df.to_csv(incorrect_mappings_csv, index=False)  # Incorrect predictions figures are saved to csv

if __name__ == "__main__":
    plot_predicted_vs_actual_by_season()
