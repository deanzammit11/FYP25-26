from pathlib import Path
from src.utils import ensure_dirs, extract_model_name_from_filename
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def season_simulation():
    results_dir = Path("data/results") # The directory containing the model results sub folders is defined
    prediction_files = [] # A list to store the model directory and 2023 prediction file path for each model is initialised

    model_dirs = [] # A list to store the paths with the model folders inside the results directory is initialised
    for folder in results_dir.iterdir(): # For each folder in the results directory
        if folder.is_dir(): # If it is a folder
            model_dirs.append(folder) # The folder path is added to the model directories list

    for model_dir in model_dirs: # For each model folder
        for prediction_file in model_dir.glob("*_2023_predictions.csv"): # For each 2023 predictions csv file inside that model directory
            prediction_files.append((model_dir, prediction_file)) # The model directory path and prediction file path are stored

    if not prediction_files: # If no prediction files were found
        raise FileNotFoundError(f"No root-level '*_2023_predictions.csv' files found in top-level model folders under {results_dir}") # A file not found error is raised

    for model_dir, prediction_file in prediction_files: # For each model directory path and model prediction csv file
        df = pd.read_csv(prediction_file) # The model predictions csv file is stored in a dataframe

        required_columns = {"Season", "HomeTeam", "AwayTeam", "ResultEncoded", "Predicted"} # The required columns are defined
        missing = required_columns.difference(df.columns) # Any missing required columns are identified and stored
        if missing: # If any required columns are missing
            raise ValueError(f"Missing required columns in {prediction_file}: {missing}") # A value error is raised

        season_df = df.dropna(subset=["Season", "HomeTeam", "AwayTeam", "ResultEncoded", "Predicted"]).copy() # Rows missing any of the required columns are removed and the result is copied
        season_df["Season"] = season_df["Season"].astype(int) # Season is converted to integer
        season_df = season_df[season_df["Season"] == 2023].copy() # Rows are filtered for rows falling under the 2023 season and are copied into a new dataframe
        season_df["ResultEncoded"] = season_df["ResultEncoded"].astype(int) # Actual outcomes are converted to integers
        season_df["Predicted"] = season_df["Predicted"].astype(int) # Predicted outcomes are converted to integers

        if season_df.empty: # If no rows remain after filtering
            raise ValueError(f"No 2023 rows found in {prediction_file}") # A value error is raised

        actual_table = {} # A dictionary which will store each club's actual league stats is initialised
        actual_head_to_head = {} # A dictionary which will store the actual head to head record between clubs is initialised
        for _, row in season_df.iterrows(): # For each actual fixture falling under the 2023 season
            home = str(row["HomeTeam"]) # The home team name is stored as a string
            away = str(row["AwayTeam"]) # The away team name is stored as a string
            result = int(row["ResultEncoded"]) # The actual outcome is stored as an integer

            if home not in actual_table: # If the home team has not yet been added to the actual table
                actual_table[home] = {"Matches Played": 0, "Wins": 0, "Draws": 0, "Losses": 0, "Points": 0} # A record with placeholder values is created for the home team
            if away not in actual_table: # If the away team has not yet been added to the actual table
                actual_table[away] = {"Matches Played": 0, "Wins": 0, "Draws": 0, "Losses": 0, "Points": 0} # A record with placeholder values is created for the away team

            actual_head_to_head.setdefault(home, {}) # If the home team does not have an entry in the actual head to head dictionary it is initialised with an empty dictionary
            actual_head_to_head.setdefault(away, {}) # If the away team does not have an entry in the actual head to head dictionary it is initialised with an empty dictionary
            actual_head_to_head[home].setdefault(away, 0) # If there is no entry for points won by the home team against the away team that entry is initialised to 0
            actual_head_to_head[away].setdefault(home, 0) # If there is no entry for points won by the away team against the home team that entry is initialised to 0

            actual_table[home]["Matches Played"] += 1 # The matches played for the home team are incremented
            actual_table[away]["Matches Played"] += 1 # The matches played for the away team are incremented

            if result == 1: # If the actual result is a home win
                actual_table[home]["Wins"] += 1 # The wins for the home team are incremented
                actual_table[away]["Losses"] += 1 # The losses for the away team are incremented
                actual_table[home]["Points"] += 3 # 3 points are added to the home team
                actual_head_to_head[home][away] += 3 # 3 head to head points for the home team against the away team are added
            elif result == 0: # If the actual result is a draw
                actual_table[home]["Draws"] += 1 # The draws for the home team are incremented
                actual_table[away]["Draws"] += 1 # The draws for the away team are incremented
                actual_table[home]["Points"] += 1 # 1 point is added to the home team
                actual_table[away]["Points"] += 1 # 1 point is added to the away team
                actual_head_to_head[home][away] += 1 # 1 head to head point for the home team against the away team is added
                actual_head_to_head[away][home] += 1 # 1 head to head point for the away team against the home team is added
            elif result == -1: # If the actual result is a home loss
                actual_table[away]["Wins"] += 1 # The wins for the away team are incremented
                actual_table[home]["Losses"] += 1 # The losses for the home team are incremented
                actual_table[away]["Points"] += 3 # 3 points are added to the away team
                actual_head_to_head[away][home] += 3 # 3 head to head points for the away team against the home team are added
            else: # If the actual result value is neither
                raise ValueError(f"Unexpected actual result value in {prediction_file}: {result}") # A value error is raised

        actual_standings = ( # An actual standings dataframe is created
            pd.DataFrame.from_dict(actual_table, orient="index") # The dictionary is converted into a table with the club names being the index
            .reset_index() # The row index is reset with the team name becoming a normal column
            .rename(columns={"index": "Club"}) # The previous index column contaning team names is renamed to Club
            .sort_values(["Points", "Club"], ascending=[False, True], kind="stable") # Teams are sorted by points in descending order and then by club name in alphabetical order
            .reset_index(drop=True) # The row index is reset with the old index column not being kept as a column unlike before
        )

        actual_ordered_groups = [] # A list to store the final actual standings after point tie breaks if there are any is initialised
        for _, group in actual_standings.groupby("Points", sort=False): # For each group after clubs are grouped based on points
            if len(group) == 1: # If only one club has that points total
                actual_ordered_groups.append(group) # The group is added
                continue # Jump to the next group in the for loop

            tied_teams = group["Club"].tolist() # A list containing the club names of the clubs tied on points is created
            group = group.copy() # The group is copied before adding the tie break columns
            head_to_head_points = [] # A list to store the head to head points for each club in the tied group is initialised
            for club in group["Club"]: # For each club tied on points
                club_head_to_head_points = 0 # A counter for the club's head to head points is initialised
                for opponent in tied_teams: # For every other tied club
                    if opponent != club: # Checks that the opponent is not the same as the club being compared
                        club_head_to_head_points += actual_head_to_head.get(club, {}).get(opponent, 0) # The points the club won against that opponent in head to head encounters are added
                head_to_head_points.append(club_head_to_head_points) # The head to head points for the current club are added to the list
            group["HeadToHeadPoints"] = head_to_head_points # A column containing the total head to head points is added
            group = group.sort_values( # The group is sorted based on head to head points
                ["HeadToHeadPoints", "Club"], # It is first sorted by head to head points and then by club name
                ascending=[False, True], # Head to head points sorting is done in descending order while sorting by club name is done in alphabetical order
                kind="stable", # Stable sorting is used so that if rows are equal on the sort keys the same previous order is kept
            ).drop(columns=["HeadToHeadPoints"]) # The head to head column is removed after group is sorted based on tie break points
            actual_ordered_groups.append(group) # The final ordered group after the tie break is added

        actual_standings = pd.concat(actual_ordered_groups, ignore_index=True) # All points groups are concatenated into one final actual standings dataframe resetting the index
        actual_standings.insert(0, "Position", range(1, len(actual_standings) + 1)) # A positions column showing the postion of the club is added as the first column
        actual_standings = actual_standings[["Position", "Club", "Matches Played", "Wins", "Draws", "Losses", "Points"]] # Columns are reordered to match the league table format

        table = {} # A dictionary which will store each club league stats is initialised
        head_to_head = {} # A dictionary which will store the head to head record between clubs is initialised

        for _, row in season_df.iterrows(): # For each predicted fixture falling udner the 2023 season
            home = str(row["HomeTeam"]) # The home team name is stored as a string
            away = str(row["AwayTeam"]) # The away team name is stored as a string
            prediction = int(row["Predicted"]) # The predicted result is stored as an integer

            if home not in table: # If the home team has not yet been added to the table
                table[home] = {"Matches Played": 0, "Wins": 0, "Draws": 0, "Losses": 0, "Points": 0} # A record with placeholder values is created for the home team
            if away not in table: # If the away team has not yet been added to the table
                table[away] = {"Matches Played": 0, "Wins": 0, "Draws": 0, "Losses": 0, "Points": 0} # A record with placeholder values is created for the away team

            head_to_head.setdefault(home, {}) # If the home team does not have an entry in the head to head dictionary it is initialised with an empty dictionary
            head_to_head.setdefault(away, {}) # If the away team does not have an entry in the head to head dictionary it is initialised with an empty dictionary
            head_to_head[home].setdefault(away, 0) # If there is no entry for points won by the home team against the away team that entry is initialised to 0
            head_to_head[away].setdefault(home, 0) # If there is no entry for points won by the away team against the home team that entry is initialised to 0

            table[home]["Matches Played"] += 1 # The matches played for the home team are incremented
            table[away]["Matches Played"] += 1 # The matches played for the away team are incremented

            if prediction == 1: # If the prediction is a home win
                table[home]["Wins"] += 1 # The wins for the home team are incremented
                table[away]["Losses"] += 1 # The losses for the away team are incremented
                table[home]["Points"] += 3 # 3 points are added to the home team
                head_to_head[home][away] += 3 # 3 head to head points for the home team against the away team are added
            elif prediction == 0: # If the prediction is a draw
                table[home]["Draws"] += 1 # The draws for the home team are incremented
                table[away]["Draws"] += 1 # The draws for the away team are incremented
                table[home]["Points"] += 1 # 1 point is added to the home team
                table[away]["Points"] += 1 # 1 point is added to the away team
                head_to_head[home][away] += 1 # 1 head to head point for the home team against the away team is added
                head_to_head[away][home] += 1 # 1 head to head point for the away team against the home team is added
            elif prediction == -1: # If the prediction is a home loss
                table[away]["Wins"] += 1 # The wins for the away team are incremented
                table[home]["Losses"] += 1 # The losses for the home team are incremented
                table[away]["Points"] += 3 # 3 points are added to the away team
                head_to_head[away][home] += 3 # 3 head to head points for the away team against the home team are added
            else: # If the prediction value is neither
                raise ValueError(f"Unexpected prediction value in {prediction_file}: {prediction}") # A value error is raised

        simulated_standings = ( # A simulated standings dataframe is created
            pd.DataFrame.from_dict(table, orient="index") # The dictionary is converted into a table with the club names being the index
            .reset_index() # The row index is reset with the team name becoming a normal column
            .rename(columns={"index": "Club"}) # The previous index column contaning team names is renamed to Club
            .sort_values(["Points", "Club"], ascending=[False, True], kind="stable") # Teams are sorted by point in descending order and then by club name in alphabetical order
            .reset_index(drop=True) # The row index is reset with the old index column not being kept as a column unlike before
        )

        ordered_groups = [] # A list to store the final standings after point tie breaks if there are any is initialised
        for _, group in simulated_standings.groupby("Points", sort=False): # For each group after clubs are grouped based on points
            if len(group) == 1: # If only one club has that points total
                ordered_groups.append(group) # The group is added
                continue # Jump to the next group in the for loop

            tied_teams = group["Club"].tolist() # A list containing the club names of the clubs tied on points is created
            group = group.copy() # The group is copied before adding the tie break columns
            head_to_head_points = [] # A list to store the head to head points for each club in the tied group is initialised
            for club in group["Club"]: # For each club tied on points
                club_head_to_head_points = 0 # A counter for the club's head to head points is initialised
                for opponent in tied_teams: # For every other tied club
                    if opponent != club: # Checks that the opponent is not the same as the club being compared
                        club_head_to_head_points += head_to_head.get(club, {}).get(opponent, 0) # The points the club won against that opponent in head to head encounters are added
                head_to_head_points.append(club_head_to_head_points) # The head to head points for the current club are added to the list
            group["HeadToHeadPoints"] = head_to_head_points # A column containing the total head to head points is added
            group = group.sort_values( # The group is sorted based on head to head points
                ["HeadToHeadPoints", "Club"], # It is first sorted by head to head points and then by club name
                ascending=[False, True], # Head to head points sorting is done in descending order while sorting by club name is done in alphabetical order
                kind="stable", # Stable sorting is used so that if rows are equal on the sort keys the same previous order is kept
            ).drop(columns=["HeadToHeadPoints"]) # The head to head column is removed after group is sorted based on tie break points
            ordered_groups.append(group) # The final ordered group after the tie break is added

        simulated_standings = pd.concat(ordered_groups, ignore_index=True) # All points groups are concatenated into one final simulated standings dataframe resetting the index
        simulated_standings.insert(0, "Position", range(1, len(simulated_standings) + 1)) # A positions column showing the postion of the club is added as the first column
        simulated_standings = simulated_standings[["Position", "Club", "Matches Played", "Wins", "Draws", "Losses", "Points"]] # Columns are reordered to match the league table format

        model_name = extract_model_name_from_filename(str(prediction_file)) # The model name is extracted from the predictions file name after it is converted to a string and is then saved
        output_image = model_dir / f"{model_name}_2023_league_table.png" # The output path for the league table image is created
        actual_output_csv = model_dir / f"{model_name}_2023_actual_league_table.csv" # The output path for the actual league table csv is created
        simulated_output_csv = model_dir / f"{model_name}_2023_simulated_league_table.csv" # The output path for the simulated league table csv is created
        comparison_output_csv = model_dir / f"{model_name}_2023_league_table_comparison.csv" # The output path for the league table comparison csv is created
        ensure_dirs(str(model_dir)) # Ensures the model directory exists before saving the image

        actual_standings.to_csv(actual_output_csv, index=False) # The actual league table csv is saved in the specified directory
        simulated_standings.to_csv(simulated_output_csv, index=False) # The simulated league table csv is saved in the specified directory

        comparison = actual_standings[["Position", "Club", "Points"]].rename(columns={"Position": "ActualPosition", "Points": "ActualPoints"}).merge(simulated_standings[["Position", "Club", "Points"]].rename(columns={"Position": "SimulatedPosition", "Points": "SimulatedPoints"}), on="Club", how="inner") # A comparison dataframe containing the actual positions, club names and points is created and then joined to the simulated standings using an inner join on the club name
        comparison["PositionDifference"] = comparison["SimulatedPosition"] - comparison["ActualPosition"] # The position difference between the simulated and actual table is calculated
        comparison = comparison.sort_values("ActualPosition").reset_index(drop=True) # The comparison dataframe is sorted by the actual position and the index is reset
        comparison.to_csv(comparison_output_csv, index=False) # The league table comparison csv is saved in the specified directory

        fig_height = max(6, 1.2 + len(simulated_standings) * 0.38) # Figure height is set to either 6 or the sum of 1.2 + the number of teams in the table multiplied by 0.38
        fig, axes = plt.subplots(1, 2, figsize=(20, fig_height)) # A matplotlib figure with two axes is created for the actual and simulated tables
        fig.suptitle(f"{model_name.replace('_', ' ').title()} - Actual vs Simulated 2023 League Tables", fontsize=16, y=0.99) # The overall figure title is set replacing an _ with a space and capitalising the first letter of every word

        for ax, standings, title in zip(axes, [actual_standings, simulated_standings], ["Actual Table", "Simulated Table"]): # Each axis is paired with one standings dataframe and the respective subtitle
            ax.axis("off") # Axis are hidden since each subplot is a table
            ax.set_title(title, fontsize=13, pad=0, y=0.94) # The subplot title is set

            table_plot = ax.table( # A matplotlib table is built using the standings dataframe
                cellText=standings.values, # Table cell values are filled up using the dataframe rows
                colLabels=standings.columns.tolist(), # Column headers are set from the dataframe column names
                loc="center", # The table is placed at the centre of the axis
                cellLoc="center", # The text inside each cell is centered
            )
            table_plot.auto_set_font_size(False) # Automatic font resizing is disabled
            table_plot.set_fontsize(10) # A fixed font size of 10 is set to the table text
            table_plot.scale(1, 1.35) # Table width is set to 1 and table height is set to 1.35

            for row_idx in range(1, len(standings) + 1): # For each row in the standings table excluding the first row containing the headers
                table_plot[(row_idx, 1)].get_text().set_ha("left") # Left aligns the club name under the club column
            for col_idx in range(len(standings.columns)): # For each column in the standings table
                table_plot[(0, col_idx)].set_text_props(weight="bold") # Bold formatting is set to the header

        plt.tight_layout() # Layout is adjusted while leaving room for the overall title
        plt.savefig(output_image, dpi=200, bbox_inches="tight") # Image is saved to the specified path setting the resolution and cropping around the table removing extra whitespace
        plt.close(fig) # Figure is closed

        print(f"Saved league table comparison image: {output_image}") # Prints confirmation message that the image has been saved
        print(f"Saved actual league table csv: {actual_output_csv}") # Prints confirmation message that the actual league table csv has been saved
        print(f"Saved simulated league table csv: {simulated_output_csv}") # Prints confirmation message that the simulated league table csv has been saved
        print(f"Saved league table comparison csv: {comparison_output_csv}") # Prints confirmation message that the comparison csv has been saved

if __name__ == "__main__":
    season_simulation()