from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.utils import ensure_dirs, extract_model_name_from_filename

def betting_simulation(max_fraction=1.0, min_edge_per_model={"logistic_regression": 0.15, "random_forest": 0.15, "xgboost": 0.1}, min_odds_per_model={"logistic_regression": 1.5, "random_forest": 2.0, "xgboost": 1.5}, starting_bankroll_values=[380.0, 760.0, 1140.0, 1520.0, 1900.0, 2280.0, 2660.0, 3040.0, 3420.0, 3800.0], prediction_years=[2020, 2021, 2022, 2023]):
    max_fraction = float(max_fraction) # Maximum allowed stake fraction is converted to float
    min_edge_per_model = {k: float(v) for k, v in min_edge_per_model.items()} # Minimum Kelly edge per model is converted to float
    min_odds_per_model = {k: float(v) for k, v in min_odds_per_model.items()} # Minimum odds per model is converted to float
    minimum_bet_stake = 0.01 # The minimum allowed bet staked is initialised
    minimum_balance_to_continue = 1.0 # The minimum balance required to continue the simulation is initialised
    fixtures_per_gameweek = 10 # The number of fixtures in a gameweek is initialised
    fixture_columns = ["Season", "Date", "HomeTeam", "AwayTeam"] # Fixture columns are defined
    probability_columns = ["ProbHome", "ProbDraw", "ProbAway"] # Model probability columns are defined
    odds_columns = ["Bet365HomeWinOdds", "Bet365DrawOdds", "Bet365AwayWinOdds"] # Odds columns are defined
    intelligent_columns = ["EloTierHome", "EloTierAway"] # Elo tier columns are defined
    starting_bankroll_values = [float(value) for value in starting_bankroll_values] # Each starting bankroll value is converted to float
    if not starting_bankroll_values: # If no starting bankroll values are provided
        raise ValueError("At least one starting bankroll value must be provided") # A value error is raised
    standard_bankroll_value = starting_bankroll_values[0] # The first starting bankroll value is used as the bankroll value for the main simulation
    training_data_path = Path("data/features/eng1_data_combined.csv") # Source path for training data is defined
    if not training_data_path.exists(): # If training data file was not found
        raise FileNotFoundError(f"Training data file not found: {training_data_path}") # A file not found error is raised
    training_data_df = pd.read_csv(training_data_path) # Training data dataframe is loaded
    required_training_columns = {"Season", "ResultEncoded", "EloTierHome", "EloTierAway"} # Required columns to build Elo tier matchup history are defined
    missing_training_columns = sorted(required_training_columns.difference(training_data_df.columns)) # Any missing required training columns are identified
    if missing_training_columns: # If any required columns are missing
        raise ValueError(f"Missing required columns in {training_data_path}: {missing_training_columns}") # A value error is raised
    training_data_df["Season"] = pd.to_numeric(training_data_df["Season"], errors="coerce") # Season column in training data is converted to numeric
    results_path = Path("data/results") # Results directory path is defined

    for prediction_year in sorted(prediction_years): # For each prediction year in ascending order
        print(f"\nRunning betting simulation for {prediction_year}...") # Confirmation message is printed
        prediction_files = sorted(results_path.glob(f"*/*_{prediction_year}_predictions.csv")) # The model prediction files for the respective year are found and sorted
        if not prediction_files: # If no prediction files were found
            raise FileNotFoundError(f"No prediction files found under {results_path}") # A file not found error is raised

        combined_summaries = [] # A list to store each model summary dataframe is initialised
        starting_bankroll_summary_rows = [] # A list to store each model, strategy, and starting bankroll summary is initialised
        training_base = training_data_df[training_data_df["Season"] < prediction_year].copy() # Only seasons before the prediction year are kept for initial matchup history
        training_base["ResultEncoded"] = pd.to_numeric(training_base["ResultEncoded"], errors="coerce") # Match outcome in training data is converted to numeric
        training_base["EloTierHome"] = pd.to_numeric(training_base["EloTierHome"], errors="coerce") # Home Elo tier in training data is converted to numeric
        training_base["EloTierAway"] = pd.to_numeric(training_base["EloTierAway"], errors="coerce") # Away Elo tier in training data is converted to numeric
        training_base = training_base.dropna(subset=["ResultEncoded", "EloTierHome", "EloTierAway"]).copy() # Training rows with any missing outcomes or Elo tiers are dropped

        for prediction_file in prediction_files: # For each model prediction file
            df = pd.read_csv(prediction_file) # Prediction csv is read into a dataframe
            model_name = extract_model_name_from_filename(str(prediction_file)) # Model name is extracted from prediction filename

            required_columns = set(fixture_columns + ["ResultEncoded"] + probability_columns + odds_columns + intelligent_columns) # All columns required for the betting simulation are defined
            missing_columns = sorted(required_columns.difference(df.columns)) # Any missing required columns are identified
            if missing_columns: # If any required columns are missing
                raise ValueError(f"Missing required columns for Kelly criterion in {prediction_file}: {missing_columns}") # A value error is raised

            df["DateParsed"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce") # Fixture date is converted to datetime
            df["ResultEncoded"] = pd.to_numeric(df["ResultEncoded"], errors="coerce") # Result in prediction rows is converted to numeric
            df["EloTierHome"] = pd.to_numeric(df["EloTierHome"], errors="coerce") # Home Elo tier in prediction rows is converted to numeric
            df["EloTierAway"] = pd.to_numeric(df["EloTierAway"], errors="coerce") # Away Elo tier in prediction rows is converted to numeric
            df = df.sort_values(["DateParsed", "HomeTeam", "AwayTeam"]).reset_index(drop=True) # Fixtures are first sorted chronologically and then by team names
            df["Gameweek"] = (df.index // fixtures_per_gameweek) + 1 # The gameweek number is to each fixture
            season_game_count = len(df) # The total number of fixtures in the prediction season is stored
            if season_game_count <= 0: # If there are no fixtures in the prediction file
                raise ValueError(f"No fixtures found in prediction file: {prediction_file}") # A value error is raised
            season_gameweek_count = int(df["Gameweek"].max()) # The total number of gameweeks in the prediction season is stored by storing the largest gameweek value
            gameweek_fixture_counts = df.groupby("Gameweek").size().to_dict() # The number of fixtures in each gameweek are counted and the groups are stored in a dictionary

            for bankroll_value in starting_bankroll_values: # For each starting bankroll value a simulation is run
                base_weekly_budget = bankroll_value / season_gameweek_count # The base weekly staking budget is set to the starting bankroll value divided by the number of gameweeks
                kelly_balance = bankroll_value # Kelly bankroll is initialised with the starting bankroll
                half_kelly_balance = bankroll_value # Half-Kelly bankroll is initialised with the starting bankroll
                intelligent_kelly_balance = bankroll_value # Intelligent Kelly bankroll is initialised with the starting bankroll
                intelligent_half_kelly_balance = bankroll_value # Intelligent Half-Kelly bankroll is initialised with the starting bankroll
                flat_prediction_balance = bankroll_value # Flat prediction bankroll is initialised with the starting bankroll
                kelly_weekly_available = base_weekly_budget # Kelly weekly staking budget is initialised
                half_kelly_weekly_available = base_weekly_budget # Half-Kelly weekly staking budget is initialised
                intelligent_kelly_weekly_available = base_weekly_budget # Intelligent Kelly weekly staking budget is initialised
                intelligent_half_kelly_weekly_available = base_weekly_budget # Intelligent Half-Kelly weekly staking budget is initialised
                flat_prediction_weekly_available = base_weekly_budget # Flat prediction weekly staking budget is initialised
                kelly_weekly_profit_loss = 0.0 # Kelly weekly profit or loss count is initialised
                half_kelly_weekly_profit_loss = 0.0 # Half-Kelly weekly profit or loss count is initialised
                intelligent_kelly_weekly_profit_loss = 0.0 # Intelligent Kelly weekly profit or loss count is initialised
                intelligent_half_kelly_weekly_profit_loss = 0.0 # Intelligent Half-Kelly weekly profit or loss count is initialised
                flat_prediction_weekly_profit_loss = 0.0 # Flat prediction weekly profit or loss count is initialised
                current_gameweek = None # The current gameweek tracker is initialised
                current_gameweek_fixture_count = 0 # The number of fixtures in the current gameweek is initialised
                flat_prediction_stake_amount = 0.0 # The flat prediction per match stake for the current gameweek is initialised
                peak_kelly = bankroll_value # Peak Kelly bankroll value is initialised
                peak_half_kelly = bankroll_value # Peak Half-Kelly bankroll value is initialised
                peak_intelligent_kelly = bankroll_value # Peak Intelligent Kelly bankroll value is initialised
                peak_intelligent_half_kelly = bankroll_value # Peak Intelligent Half-Kelly bankroll value is initialised
                peak_flat_prediction = bankroll_value # Peak flat prediction bankroll value is initialised
                kelly_stopped_early = False # Flag indicating whether Kelly stopped early is initialised
                kelly_stop_date = "" # The fixture date when Kelly stopped early is initialised
                half_kelly_stopped_early = False # Flag indicating whether Half-Kelly stopped early is initialised
                half_kelly_stop_date = "" # The fixture date when Half-Kelly stopped early is initialised
                intelligent_kelly_stopped_early = False # Flag indicating whether Intelligent Kelly stopped early is initialised
                intelligent_kelly_stop_date = "" # The fixture date when Intelligent Kelly stopped early is initialised
                intelligent_half_kelly_stopped_early = False # Flag indicating whether Intelligent Half-Kelly stopped early is initialised
                intelligent_half_kelly_stop_date = "" # The fixture date when Intelligent Half-Kelly stopped early is initialised
                flat_prediction_stopped_early = False # Flag indicating whether the flat prediction strategy stopped early is initialised
                flat_prediction_stop_date = "" # The fixture date when the flat prediction strategy stopped early is initialised
                rows = [] # An empty list which will store a dictionary for each fixture containing the data related to the bet on that fixture is intialised
                matchup_totals = {} # An empty dictionary which will store the total number of historical matches for each Elo tier home vs away pairing is initialised
                matchup_outcome_counts = {} # An empty dictionary which will store the total historical outcome counts for each Elo tier home vs away pairing is initialised

                for _, training_row in training_base.iterrows(): # For each training set fixture
                    home_tier = int(training_row["EloTierHome"]) # Home Elo tier is converted to integer
                    away_tier = int(training_row["EloTierAway"]) # Away Elo tier is converted to integer
                    outcome = int(training_row["ResultEncoded"]) # Match outcome is converted to integer
                    tier_key = (home_tier, away_tier) # Elo tier matchup key is created and stored
                    outcome_key = (home_tier, away_tier, outcome) # Elo tier matchup and outcome key is created and stored
                    matchup_totals[tier_key] = matchup_totals.get(tier_key, 0) + 1 # The elo tier key is looked up and if it is found it is returned and incremented and if it is not found meaning it is the first appearance it is intialised to 0 and incremented
                    matchup_outcome_counts[outcome_key] = matchup_outcome_counts.get(outcome_key, 0) + 1 # The elo tier and outcome key is looked up and if it is found it is returned and incremented and if it is not found meaning it is the first appearance it is intialised to 0 and incremented

                for _, row in df.iterrows(): # For each predicted fixture in chronological order
                    fixture_gameweek = int(row["Gameweek"]) # The gameweek number for the current fixture is stored as an integer
                    if current_gameweek is None: # If this is the first fixture
                        current_gameweek = fixture_gameweek # The current gameweek tracker is initialised with the value of the gameweek for the respective fixture
                        current_gameweek_fixture_count = int(gameweek_fixture_counts.get(current_gameweek, 0)) # The number of fixtures in the first gameweek is stored
                        flat_prediction_stake_amount = (flat_prediction_weekly_available / current_gameweek_fixture_count) if current_gameweek_fixture_count > 0 else 0.0 # The flat prediction per match stake is set from the current gameweek budget divided by the number of fixture in the current gameweek
                    elif fixture_gameweek != current_gameweek: # If a new gameweek has started
                        kelly_weekly_available = max(0.0, base_weekly_budget + kelly_weekly_available + kelly_weekly_profit_loss) # Kelly weekly budget is set to the base weekly budget plus the unused budget from the previous gameweek plus the profit or loss from the previous gameweek
                        half_kelly_weekly_available = max(0.0, base_weekly_budget + half_kelly_weekly_available + half_kelly_weekly_profit_loss) # Half-Kelly weekly budget is set to the base weekly budget plus the unused budget from the previous gameweek plus the profit or loss from the previous gameweek
                        intelligent_kelly_weekly_available = max(0.0, base_weekly_budget + intelligent_kelly_weekly_available + intelligent_kelly_weekly_profit_loss) # Intelligent Kelly weekly budget is set to the base weekly budget plus the unused budget from the previous gameweek plus the profit or loss from the previous gameweek
                        intelligent_half_kelly_weekly_available = max(0.0, base_weekly_budget + intelligent_half_kelly_weekly_available + intelligent_half_kelly_weekly_profit_loss) # Intelligent Half-Kelly weekly budget is set to the base weekly budget plus the unused budget from the previous gameweek plus the profit or loss from the previous gameweek
                        flat_prediction_weekly_available = max(0.0, base_weekly_budget + flat_prediction_weekly_available + flat_prediction_weekly_profit_loss) # Flat prediction weekly budget is set to the base weekly budget plus the unused budget from the previous gameweek plus the profit or loss from the previous gameweek
                        kelly_weekly_profit_loss = 0.0 # Kelly weekly profit or loss count is reset for the new gameweek
                        half_kelly_weekly_profit_loss = 0.0 # Half-Kelly weekly profit or loss count is reset for the new gameweek
                        intelligent_kelly_weekly_profit_loss = 0.0 # Intelligent Kelly weekly profit or loss count is reset for the new gameweek
                        intelligent_half_kelly_weekly_profit_loss = 0.0 # Intelligent Half-Kelly weekly profit or loss count is reset for the new gameweek
                        flat_prediction_weekly_profit_loss = 0.0 # Flat prediction weekly profit or loss count is reset for the new gameweek
                        current_gameweek = fixture_gameweek # The current gameweek tracker is updated with the value of the current gameweek
                        current_gameweek_fixture_count = int(gameweek_fixture_counts.get(current_gameweek, 0)) # The number of fixtures in the new current gameweek is stored
                        flat_prediction_stake_amount = (flat_prediction_weekly_available / current_gameweek_fixture_count) if current_gameweek_fixture_count > 0 else 0.0 # The flat prediction per match stake is updated for the new gameweek by dividing the current week's budget divided by its fixture count

                    predicted_outcome = int(row["Predicted"]) # The model's predicted outcome for the current fixture is stored
                    prediction_option_map = { # A dictionary which maps the model prediction to the respective outcome label, probability and odds columns is built
                        1: ("Home", row["ProbHome"], row["Bet365HomeWinOdds"]), # Home option
                        0: ("Draw", row["ProbDraw"], row["Bet365DrawOdds"]), # Draw option
                        -1: ("Away", row["ProbAway"], row["Bet365AwayWinOdds"]), # Away option
                    }
                    if predicted_outcome not in prediction_option_map: # If the prediction label is not in the prediction options dictionary mapping
                        raise ValueError(f"Unsupported predicted outcome {predicted_outcome} for fixture: {row['Date']} {row['HomeTeam']} vs {row['AwayTeam']}") # A value error is raised

                    label, probability, odds = prediction_option_map[predicted_outcome] # The labl, probability and odds for the model prediction are stored
                    probability = float(probability) # Option probability is converted to float
                    q = 1.0 - probability # Option loss probability is calculated
                    b = float(odds) - 1.0 # Net odds are calculated from odds
                    best = None # Best bet candidate for the current fixture is initialised as None
                    if float(odds) < min_odds_per_model.get(model_name, 1.0): # If the odds are less than the minimum odds required for the respective model to place a bet then no bet is placed
                        pass # Exits the if statement
                    elif b > 0: # If net odds are valid
                        edge = probability - (q / b) # Kelly edge is calculated
                        if edge > min_edge_per_model.get(model_name, 0.0): # If the edge is greater than the minimum edge required for the respective model to place a bet then the bet is placed
                            best = { # Bet dictionary for the predicted outcome is built
                                "BetOn": label, # Bet label is stored
                                "BetOnEncoded": predicted_outcome, # Encoded bet result is stored
                                "p": probability, # Option win probability is stored
                                "q": q, # Option loss probability is stored
                                "b": b, # Net odds are stored
                                "Odds": float(odds), # Odds are stored
                                "KellyFraction": max(0.0, edge), # Kelly fraction is stored
                            }

                    kelly_before = kelly_balance # Kelly bankroll before fixture is stored
                    half_kelly_before = half_kelly_balance # Half-Kelly bankroll before fixture is stored
                    intelligent_kelly_before = intelligent_kelly_balance # Intelligent Kelly bankroll before fixture is stored
                    intelligent_half_kelly_before = intelligent_half_kelly_balance # Intelligent Half-Kelly bankroll before fixture is stored
                    flat_prediction_before = flat_prediction_balance # Flat prediction bankroll before fixture is stored
                    home_tier_value = int(row["EloTierHome"]) if pd.notna(row["EloTierHome"]) else pd.NA # Current fixture home Elo tier is captured and if not found it is set to NA
                    away_tier_value = int(row["EloTierAway"]) if pd.notna(row["EloTierAway"]) else pd.NA # Current fixture away Elo tier is captured and if not found it is set to NA

                    base_fixture_row = { # Base fixture fields present in each row are stored in a dictionary
                        "Model": model_name, # Model name is stored
                        "Date": row["Date"], # Fixture date is stored
                        "HomeTeam": row["HomeTeam"], # Home team is stored
                        "AwayTeam": row["AwayTeam"], # Away team is stored
                        "Gameweek": fixture_gameweek, # Gameweek number is stored
                        "BetPlaced": bool(best is not None), # Flag indicating whether bet is placed is stored
                        "ModelPredictedOutcome": label, # The model predicted outcome label is stored
                        "ModelPredictedEncoded": predicted_outcome, # The model predicted encoded outcome is stored
                        "ActualResult": int(row["ResultEncoded"]), # Actual outcome is stored
                        "EloTierHome": home_tier_value, # Home Elo tier is stored
                        "EloTierAway": away_tier_value, # Away Elo tier is stored
                    }

                    flat_prediction_stake = min(flat_prediction_stake_amount, flat_prediction_weekly_available) if flat_prediction_before >= flat_prediction_stake_amount and flat_prediction_weekly_available > 0.0 else 0.0 # The flat prediction stake is set to the smaller of the planned flat stake and the remaining weekly budget if the bankroll can cover the stake and weekly budget is still available, otherwise it sets the stake to 0.0
                    flat_prediction_won = int(row["ResultEncoded"]) == predicted_outcome # It is checked if the model prediction matches the actual result
                    flat_prediction_profit_loss = flat_prediction_stake * b if flat_prediction_won else -flat_prediction_stake # Flat prediction profit or loss is calculated
                    flat_prediction_balance = flat_prediction_before + flat_prediction_profit_loss # Flat prediction bankroll is updated
                    flat_prediction_weekly_available = max(0.0, flat_prediction_weekly_available - flat_prediction_stake) # Flat prediction weekly available budget is reduced by the current stake
                    flat_prediction_weekly_profit_loss += flat_prediction_profit_loss # Flat prediction weekly profit or loss for the current gameweek is updated
                    if (not flat_prediction_stopped_early) and flat_prediction_before >= minimum_balance_to_continue and flat_prediction_balance < minimum_balance_to_continue: # If stopped early flag is false, the balance before the fixture was greater than the balance required to continue and after the fixture the balance fell below the minimum balance required to continue
                        flat_prediction_stopped_early = True # Flag indicating early stopping is set to true
                        flat_prediction_stop_date = str(row["Date"]) # The date of the fixture which resulted in the early stopping is saved
                    peak_flat_prediction = max(peak_flat_prediction, flat_prediction_balance) # Peak flat prediction bankroll is updated

                    if best is None: # If no good bet was found
                        rows.append({ # No bet fixture row is appended
                            **base_fixture_row, # The base fixture fields are stored
                            "BetOn": "", # Bet label is left empty because no bet is placed and stored
                            "BetOnEncoded": pd.NA, # Encoded bet outcome is set to missing because no bet is placed and stored
                            "p": pd.NA, # Win probability is set to missing because no bet is placed and stored
                            "q": pd.NA, # Loss probability is set to missing because no bet is placed and stored
                            "b": pd.NA, # Net odds are set to missing because no bet is placed and stored
                            "Odds": pd.NA, # Odds are set to missing because no bet is placed and stored
                            "KellyFraction": 0.0, # Kelly stake fraction is set to zero because no bet is placed and stored
                            "HalfKellyFraction": 0.0, # Half-Kelly stake fraction is set to zero because no bet is placed and stored
                            "KellyStake": 0.0, # Kelly stake amount is set to zero because no bet is placed and stored
                            "HalfKellyStake": 0.0, # Half-Kelly stake amount is set to zero because no bet is placed and stored
                            "KellyProfitLoss": 0.0, # Kelly profit or loss is zero because no bet is placed and stored
                            "HalfKellyProfitLoss": 0.0, # Half-Kelly profit or loss is zero because no bet is placed and stored
                            "IntelligentMultiplier": pd.NA, # Intelligent multiplier is missing because no bet is placed and stored
                            "IntelligentKellyFraction": 0.0, # Intelligent Kelly fraction is set to zero because no bet is placed and stored
                            "IntelligentHalfKellyFraction": 0.0, # Intelligent Half-Kelly fraction is set to zero because no bet is placed and stored
                            "IntelligentKellyStake": 0.0, # Intelligent Kelly stake amount is set to zero because no bet is placed and stored
                            "IntelligentHalfKellyStake": 0.0, # Intelligent Half-Kelly stake amount is set to zero because no bet is placed and stored
                            "IntelligentKellyProfitLoss": 0.0, # Intelligent Kelly profit or loss is zero because no bet is placed and stored
                            "IntelligentHalfKellyProfitLoss": 0.0, # Intelligent Half-Kelly profit or loss is zero because no bet is placed and stored
                            "FlatPredictionStake": flat_prediction_stake, # Flat prediction stake amount is stored
                            "FlatPredictionProfitLoss": flat_prediction_profit_loss, # Flat prediction profit or loss is stored
                            "KellyBalanceBefore": kelly_before, # Kelly bankroll before fixture is stored
                            "KellyBalanceAfter": kelly_balance, # Kelly bankroll after fixture is stored
                            "HalfKellyBalanceBefore": half_kelly_before, # Half-Kelly bankroll before fixture is stored
                            "HalfKellyBalanceAfter": half_kelly_balance, # Half-Kelly bankroll after fixture is stored
                            "IntelligentKellyBalanceBefore": intelligent_kelly_before, # Intelligent Kelly bankroll before fixture is stored
                            "IntelligentKellyBalanceAfter": intelligent_kelly_balance, # Intelligent Kelly bankroll after fixture is stored
                            "IntelligentHalfKellyBalanceBefore": intelligent_half_kelly_before, # Intelligent Half-Kelly bankroll before fixture is stored
                            "IntelligentHalfKellyBalanceAfter": intelligent_half_kelly_balance, # Intelligent Half-Kelly bankroll after fixture is stored
                            "FlatPredictionBalanceBefore": flat_prediction_before, # Flat prediction bankroll before fixture is stored
                            "FlatPredictionBalanceAfter": flat_prediction_balance, # Flat prediction bankroll after fixture is stored
                            "KellyDropFromPeak": (peak_kelly - kelly_balance) / peak_kelly if peak_kelly > 0 else 0.0, # Kelly drop from peak after fixture is stored
                            "HalfKellyDropFromPeak": (peak_half_kelly - half_kelly_balance) / peak_half_kelly if peak_half_kelly > 0 else 0.0, # Half-Kelly drop from peak after fixture is stored
                            "IntelligentKellyDropFromPeak": (peak_intelligent_kelly - intelligent_kelly_balance) / peak_intelligent_kelly if peak_intelligent_kelly > 0 else 0.0, # Intelligent Kelly drop from peak after fixture is stored
                            "IntelligentHalfKellyDropFromPeak": (peak_intelligent_half_kelly - intelligent_half_kelly_balance) / peak_intelligent_half_kelly if peak_intelligent_half_kelly > 0 else 0.0, # Intelligent Half-Kelly drop from peak after fixture is stored
                            "FlatPredictionDropFromPeak": (peak_flat_prediction - flat_prediction_balance) / peak_flat_prediction if peak_flat_prediction > 0 else 0.0, # Flat prediction drop from peak after fixture is stored
                        })
                    else: # If a good bet is identified
                        kelly_fraction = min(max_fraction, max(0.0, float(best["KellyFraction"]))) # Kelly fraction is set where if it is negative it is set to 0 and if it is larger than the max fraction the max fraction is taken
                        half_kelly_fraction = min(max_fraction, max(0.0, 0.5 * float(best["KellyFraction"]))) # Half-Kelly fraction is set where if it is negative it is set to 0 and if it is larger than the max fraction the max fraction is taken

                        if pd.isna(home_tier_value) or pd.isna(away_tier_value): # If Elo tiers are missing
                            raise ValueError(f"Missing Elo tier data for fixture: {row['Date']} {row['HomeTeam']} vs {row['AwayTeam']}") # A value error is raised
                        tier_key = (int(home_tier_value), int(away_tier_value)) # Elo tier matchup key is created and stored
                        outcome_key = (int(home_tier_value), int(away_tier_value), int(best["BetOnEncoded"])) # Elo tier matchup and predicted outcome key is created and stored
                        historical_total = matchup_totals.get(tier_key, 0) # Historical total matches for the tier matchup are read
                        if historical_total <= 0: # If no historical matches exist for this Elo tier matchup
                            raise ValueError(f"Insufficient Elo tier matchup history for fixture: {row['Date']} {row['HomeTeam']} vs {row['AwayTeam']} (home tier={home_tier_value}, away tier={away_tier_value})") # A value error is raised
                        intelligent_multiplier = matchup_outcome_counts.get(outcome_key, 0) / historical_total # Intelligent multiplier is calculated

                        intelligent_kelly_fraction = min(max_fraction, max(0.0, kelly_fraction * intelligent_multiplier)) # Intelligent Kelly fraction is set where if it is negative it is set to 0 and if it is larger than the max fraction the max fraction is taken
                        intelligent_half_kelly_fraction = min(max_fraction, max(0.0, half_kelly_fraction * intelligent_multiplier)) # Intelligent Half-Kelly fraction is set where if it is negative it is set to 0 and if it is larger than the max fraction the max fraction is taken

                        kelly_stake = min(kelly_before * kelly_fraction, kelly_weekly_available) if kelly_before >= minimum_balance_to_continue and kelly_weekly_available > 0.0 else 0.0 # Kelly stake is set to the smaller of the Kelly stake and the remaining weekly budget if the bankroll can cover the stake and weekly budget is still available, otherwise it sets the stake to 0.0
                        half_kelly_stake = min(half_kelly_before * half_kelly_fraction, half_kelly_weekly_available) if half_kelly_before >= minimum_balance_to_continue and half_kelly_weekly_available > 0.0 else 0.0 # Half-Kelly stake is set to the smaller of the Half-Kelly stake and the remaining weekly budget if the bankroll can cover the stake and weekly budget is still available, otherwise it sets the stake to 0.0
                        intelligent_kelly_stake = min(intelligent_kelly_before * intelligent_kelly_fraction, intelligent_kelly_weekly_available) if intelligent_kelly_before >= minimum_balance_to_continue and intelligent_kelly_weekly_available > 0.0 else 0.0 # Intelligent Kelly stake is set to the smaller of the Intelligent Kelly stake and the remaining weekly budget if the bankroll can cover the stake and weekly budget is still available, otherwise it sets the stake to 0.0
                        intelligent_half_kelly_stake = min(intelligent_half_kelly_before * intelligent_half_kelly_fraction, intelligent_half_kelly_weekly_available) if intelligent_half_kelly_before >= minimum_balance_to_continue and intelligent_half_kelly_weekly_available > 0.0 else 0.0 # Intelligent Half-Kelly stake is set to the smaller of the Intelligent Half-Kelly stake and the remaining weekly budget if the bankroll can cover the stake and weekly budget is still available, otherwise it sets the stake to 0.0

                        if 0.0 < kelly_stake <= minimum_bet_stake: # If the Kelly stake is positive but less than or equal to the minimum allowed stake
                            kelly_fraction = 0.0 # The Kelly stake fraction is set to 0
                            kelly_stake = 0.0 # The Kelly stake is set to 0
                        if 0.0 < half_kelly_stake <= minimum_bet_stake: # If the Half-Kelly stake is positive but less than or equal to the minimum allowed stake
                            half_kelly_fraction = 0.0 # The Half-Kelly stake fraction is set to 0
                            half_kelly_stake = 0.0 # The Half-Kelly stake is set to 0
                        if 0.0 < intelligent_kelly_stake <= minimum_bet_stake: # If the Intelligent Kelly stake is positive but less than or equal to the minimum allowed stake
                            intelligent_kelly_fraction = 0.0 # The Intelligent Kelly stake fraction is set to 0
                            intelligent_kelly_stake = 0.0 # The Intelligent Kelly stake is set to 0
                        if 0.0 < intelligent_half_kelly_stake <= minimum_bet_stake: # If the Intelligent Half-Kelly stake is positive but less than or equal to the minimum allowed stake
                            intelligent_half_kelly_fraction = 0.0 # The Intelligent Half-Kelly stake fraction is set to 0
                            intelligent_half_kelly_stake = 0.0 # The Intelligent Half-Kelly stake is set to 0

                        bet_placed = any(stake > 0.0 for stake in [kelly_stake, half_kelly_stake, intelligent_kelly_stake, intelligent_half_kelly_stake]) # If at least one strategy placed a valid stake on the fixture bet_placed is set to true and false otherwise

                        won = int(row["ResultEncoded"]) == int(best["BetOnEncoded"]) # It is checked if bet won
                        kelly_profit_loss = kelly_stake * best["b"] if won else -kelly_stake # Kelly profit or loss is calculated
                        half_kelly_profit_loss = half_kelly_stake * best["b"] if won else -half_kelly_stake # Half-Kelly profit or loss is calculated
                        intelligent_kelly_profit_loss = intelligent_kelly_stake * best["b"] if won else -intelligent_kelly_stake # Intelligent Kelly profit or loss is calculated
                        intelligent_half_kelly_profit_loss = intelligent_half_kelly_stake * best["b"] if won else -intelligent_half_kelly_stake # Intelligent Half-Kelly profit or loss is calculated

                        kelly_weekly_available = max(0.0, kelly_weekly_available - kelly_stake) # Kelly weekly available budget is reduced by the current stake
                        half_kelly_weekly_available = max(0.0, half_kelly_weekly_available - half_kelly_stake) # Half-Kelly weekly available budget is reduced by the current stake
                        intelligent_kelly_weekly_available = max(0.0, intelligent_kelly_weekly_available - intelligent_kelly_stake) # Intelligent Kelly weekly available budget is reduced by the current stake
                        intelligent_half_kelly_weekly_available = max(0.0, intelligent_half_kelly_weekly_available - intelligent_half_kelly_stake) # Intelligent Half-Kelly weekly available budget is reduced by the current stake
                        kelly_weekly_profit_loss += kelly_profit_loss # Kelly weekly profit or loss for the current gameweek is updated
                        half_kelly_weekly_profit_loss += half_kelly_profit_loss # Half-Kelly weekly profit or loss for the current gameweek is updated
                        intelligent_kelly_weekly_profit_loss += intelligent_kelly_profit_loss # Intelligent Kelly weekly profit or loss for the current gameweek is updated
                        intelligent_half_kelly_weekly_profit_loss += intelligent_half_kelly_profit_loss # Intelligent Half-Kelly weekly profit or loss for the current gameweek is updated

                        kelly_balance = kelly_before + kelly_profit_loss # Kelly bankroll is updated
                        half_kelly_balance = half_kelly_before + half_kelly_profit_loss # Half-Kelly bankroll is updated
                        intelligent_kelly_balance = intelligent_kelly_before + intelligent_kelly_profit_loss # Intelligent Kelly bankroll is updated
                        intelligent_half_kelly_balance = intelligent_half_kelly_before + intelligent_half_kelly_profit_loss # Intelligent Half-Kelly bankroll is updated
                        if (not kelly_stopped_early) and kelly_before >= minimum_balance_to_continue and kelly_balance < minimum_balance_to_continue: # If stopped early flag is false, the balance before the fixture was greater than the balance required to continue and after the fixture the balance fell below the minimum balance required to continue
                            kelly_stopped_early = True # Flag indicating early stopping is set to true
                            kelly_stop_date = str(row["Date"]) # The date of the fixture which resulted in the early stopping is saved
                        if (not half_kelly_stopped_early) and half_kelly_before >= minimum_balance_to_continue and half_kelly_balance < minimum_balance_to_continue: # If stopped early flag is false, the balance before the fixture was greater than the balance required to continue and after the fixture the balance fell below the minimum balance required to continue
                            half_kelly_stopped_early = True # Flag indicating early stopping is set to true
                            half_kelly_stop_date = str(row["Date"]) # The date of the fixture which resulted in the early stopping is saved
                        if (not intelligent_kelly_stopped_early) and intelligent_kelly_before >= minimum_balance_to_continue and intelligent_kelly_balance < minimum_balance_to_continue: # If stopped early flag is false, the balance before the fixture was greater than the balance required to continue and after the fixture the balance fell below the minimum balance required to continue
                            intelligent_kelly_stopped_early = True # Flag indicating early stopping is set to true
                            intelligent_kelly_stop_date = str(row["Date"]) # The date of the fixture which resulted in the early stopping is saved
                        if (not intelligent_half_kelly_stopped_early) and intelligent_half_kelly_before >= minimum_balance_to_continue and intelligent_half_kelly_balance < minimum_balance_to_continue: # If stopped early flag is false, the balance before the fixture was greater than the balance required to continue and after the fixture the balance fell below the minimum balance required to continue
                            intelligent_half_kelly_stopped_early = True # Flag indicating early stopping is set to true
                            intelligent_half_kelly_stop_date = str(row["Date"]) # The date of the fixture which resulted in the early stopping is saved
                        peak_kelly = max(peak_kelly, kelly_balance) # Peak Kelly bankroll is updated
                        peak_half_kelly = max(peak_half_kelly, half_kelly_balance) # Peak Half-Kelly bankroll is updated
                        peak_intelligent_kelly = max(peak_intelligent_kelly, intelligent_kelly_balance) # Peak Intelligent Kelly bankroll is updated
                        peak_intelligent_half_kelly = max(peak_intelligent_half_kelly, intelligent_half_kelly_balance) # Peak Intelligent Half-Kelly bankroll is updated

                        rows.append({ # Bet fixture row is appended
                            **base_fixture_row, # The base fixture fields are stored
                            "BetPlaced": bet_placed, # Flag indicating whether at least one strategy placed a stake on the fixture is stored
                            "BetOn": best["BetOn"], # Selected bet label is stored
                            "BetOnEncoded": best["BetOnEncoded"], # Selected encoded outcome is stored
                            "p": best["p"], # Selected outcome win probability is stored
                            "q": best["q"], # Selected outcome loss probability is stored
                            "b": best["b"], # Selected outcome odds are stored
                            "Odds": best["Odds"], # Selected outcome odds are stored
                            "KellyFraction": kelly_fraction, # Kelly stake fraction is stored
                            "HalfKellyFraction": half_kelly_fraction, # Half-Kelly stake fraction is stored
                            "KellyStake": kelly_stake, # Kelly stake amount is stored
                            "HalfKellyStake": half_kelly_stake, # Half-Kelly stake amount is stored
                            "KellyProfitLoss": kelly_profit_loss, # Kelly profit or loss is stored
                            "HalfKellyProfitLoss": half_kelly_profit_loss, # Half-Kelly profit or loss is stored
                            "IntelligentMultiplier": intelligent_multiplier, # Intelligent Elo tier multiplier is stored
                            "IntelligentKellyFraction": intelligent_kelly_fraction, # Intelligent Kelly fraction is stored
                            "IntelligentHalfKellyFraction": intelligent_half_kelly_fraction, # Intelligent Half-Kelly fraction is stored
                            "IntelligentKellyStake": intelligent_kelly_stake, # Intelligent Kelly stake amount is stored
                            "IntelligentHalfKellyStake": intelligent_half_kelly_stake, # Intelligent Half-Kelly stake amount is stored
                            "IntelligentKellyProfitLoss": intelligent_kelly_profit_loss, # Intelligent Kelly profit or loss is stored
                            "IntelligentHalfKellyProfitLoss": intelligent_half_kelly_profit_loss, # Intelligent Half-Kelly profit or loss is stored
                            "FlatPredictionStake": flat_prediction_stake, # Flat prediction stake amount is stored
                            "FlatPredictionProfitLoss": flat_prediction_profit_loss, # Flat prediction profit or loss is stored
                            "KellyBalanceBefore": kelly_before, # Kelly bankroll before fixture is stored
                            "KellyBalanceAfter": kelly_balance, # Kelly bankroll after fixture is stored
                            "HalfKellyBalanceBefore": half_kelly_before, # Half-Kelly bankroll before fixture is stored
                            "HalfKellyBalanceAfter": half_kelly_balance, # Half-Kelly bankroll after fixture is stored
                            "IntelligentKellyBalanceBefore": intelligent_kelly_before, # Intelligent Kelly bankroll before fixture is stored
                            "IntelligentKellyBalanceAfter": intelligent_kelly_balance, # Intelligent Kelly bankroll after fixture is stored
                            "IntelligentHalfKellyBalanceBefore": intelligent_half_kelly_before, # Intelligent Half-Kelly bankroll before fixture is stored
                            "IntelligentHalfKellyBalanceAfter": intelligent_half_kelly_balance, # Intelligent Half-Kelly bankroll after fixture is stored
                            "FlatPredictionBalanceBefore": flat_prediction_before, # Flat prediction bankroll before fixture is stored
                            "FlatPredictionBalanceAfter": flat_prediction_balance, # Flat prediction bankroll after fixture is stored
                            "KellyDropFromPeak": (peak_kelly - kelly_balance) / peak_kelly if peak_kelly > 0 else 0.0, # Kelly drop from peak after fixture is stored
                            "HalfKellyDropFromPeak": (peak_half_kelly - half_kelly_balance) / peak_half_kelly if peak_half_kelly > 0 else 0.0, # Half-Kelly drop from peak after fixture is stored
                            "IntelligentKellyDropFromPeak": (peak_intelligent_kelly - intelligent_kelly_balance) / peak_intelligent_kelly if peak_intelligent_kelly > 0 else 0.0, # Intelligent Kelly drop from peak after fixture is stored
                            "IntelligentHalfKellyDropFromPeak": (peak_intelligent_half_kelly - intelligent_half_kelly_balance) / peak_intelligent_half_kelly if peak_intelligent_half_kelly > 0 else 0.0, # Intelligent Half-Kelly drop from peak after fixture is stored
                            "FlatPredictionDropFromPeak": (peak_flat_prediction - flat_prediction_balance) / peak_flat_prediction if peak_flat_prediction > 0 else 0.0, # Flat prediction drop from peak after fixture is stored
                        })

                        if all(balance < minimum_balance_to_continue for balance in [kelly_balance, half_kelly_balance, intelligent_kelly_balance, intelligent_half_kelly_balance, flat_prediction_balance]) and all(available <= 0.0 for available in [kelly_weekly_available, half_kelly_weekly_available, intelligent_kelly_weekly_available, intelligent_half_kelly_weekly_available, flat_prediction_weekly_available]): # If every bankroll is less than the minimum balance required to continue and no strategy has any weekly stake budget left
                            break # The simulation is stopped

                    if pd.notna(home_tier_value) and pd.notna(away_tier_value) and pd.notna(row["ResultEncoded"]): # If current fixture Elo tiers and actual outcome are available
                        tier_key = (int(home_tier_value), int(away_tier_value)) # Elo tier matchup key is created
                        outcome_key = (int(home_tier_value), int(away_tier_value), int(row["ResultEncoded"])) # Elo tier matchup and actual outcome key is created and stored
                        matchup_totals[tier_key] = matchup_totals.get(tier_key, 0) + 1 # Elo tier matchup pair total count is incremented
                        matchup_outcome_counts[outcome_key] = matchup_outcome_counts.get(outcome_key, 0) + 1 # Elo tier matchup pair outcome count is incremented

                betting_log = pd.DataFrame(rows) # Fixture rows are converted into a dataframe
                kelly_bets_placed = int((betting_log["KellyStake"] > 0).sum()) if not betting_log.empty else 0 # The KellyStake value is checked for each row and if it is greater than 0 it is set to True and the number of True instances are then counted
                half_kelly_bets_placed = int((betting_log["HalfKellyStake"] > 0).sum()) if not betting_log.empty else 0 # The HalfKellyStake value is checked for each row and if it is greater than 0 it is set to True and the number of True instances are then counted
                intelligent_kelly_bets_placed = int((betting_log["IntelligentKellyStake"] > 0).sum()) if not betting_log.empty else 0 # The IntelligentKellyStake value is checked for each row and if it is greater than 0 it is set to True and the number of True instances are then counted
                intelligent_half_kelly_bets_placed = int((betting_log["IntelligentHalfKellyStake"] > 0).sum()) if not betting_log.empty else 0 # The IntelligentHalfKellyStake value is checked for each row and if it is greater than 0 it is set to True and the number of True instances are then counted
                flat_prediction_bets_placed = int((betting_log["FlatPredictionStake"] > 0).sum()) if not betting_log.empty else 0 # The FlatPredictionStake value is checked for each row and if it is greater than 0 it is set to True and the number of True instances are then counted

                kelly_profit = float(kelly_balance - bankroll_value) # Total Kelly profit is calculated
                half_kelly_profit = float(half_kelly_balance - bankroll_value) # Total Half-Kelly profit is calculated
                intelligent_kelly_profit = float(intelligent_kelly_balance - bankroll_value) # Total Intelligent Kelly profit is calculated
                intelligent_half_kelly_profit = float(intelligent_half_kelly_balance - bankroll_value) # Total Intelligent Half-Kelly profit is calculated
                flat_prediction_profit = float(flat_prediction_balance - bankroll_value) # Total flat prediction profit is calculated
                kelly_total_staked = float(betting_log["KellyStake"].sum()) if not betting_log.empty else 0.0 # Total Kelly staked amount is calculated
                half_kelly_total_staked = float(betting_log["HalfKellyStake"].sum()) if not betting_log.empty else 0.0 # Total Half-Kelly staked amount is calculated
                intelligent_kelly_total_staked = float(betting_log["IntelligentKellyStake"].sum()) if not betting_log.empty else 0.0 # Total Intelligent Kelly staked amount is calculated
                intelligent_half_kelly_total_staked = float(betting_log["IntelligentHalfKellyStake"].sum()) if not betting_log.empty else 0.0 # Total Intelligent Half-Kelly staked amount is calculated
                flat_prediction_total_staked = float(betting_log["FlatPredictionStake"].sum()) if not betting_log.empty else 0.0 # Total flat prediction staked amount is calculated

                summary = pd.DataFrame([ # A summary dataframe is built
                    { # Kelly summary row is added
                        "Model": model_name, # Model name is stored in summary row
                        "Strategy": "Kelly", # Strategy label for summary row is stored
                        "InitialBankroll": bankroll_value, # Initial bankroll used for strategy is stored
                        "FinalBalance": float(kelly_balance), # Final Kelly bankroll is stored
                        "Profit": kelly_profit, # Kelly profit is stored
                        "ProfitLossPercent": (kelly_profit / bankroll_value) * 100.0 if bankroll_value > 0 else 0.0, # Kelly profit or loss percentage is stored
                        "BetsPlaced": kelly_bets_placed, # Number of fixtures where Kelly placed a bet is stored
                        "TotalStaked": kelly_total_staked, # Total Kelly staked amount is stored
                        "ROI_On_Stake": (kelly_profit / kelly_total_staked) if kelly_total_staked > 0 else 0.0, # Kelly ROI on staked amount is stored
                        "Return_On_Bankroll": (kelly_profit / bankroll_value) if bankroll_value > 0 else 0.0, # Kelly return on starting bankroll is stored
                        "MaxDropFromPeak": float(betting_log["KellyDropFromPeak"].max()) if not betting_log.empty else 0.0, # Maximum Kelly drop from peak is stored
                        "StoppedEarly": kelly_stopped_early, # Flag indicating whether Kelly stopped early is stored
                        "StopDate": kelly_stop_date, # The date of the fixture when Kelly stopped early is stored
                    },
                    { # Half-Kelly summary row is added
                        "Model": model_name, # Model name is stored in summary row
                        "Strategy": "Half-Kelly", # Strategy label for summary row is stored
                        "InitialBankroll": bankroll_value, # Initial bankroll used for strategy is stored
                        "FinalBalance": float(half_kelly_balance), # Final Half-Kelly bankroll is stored
                        "Profit": half_kelly_profit, # Half-Kelly profit is stored
                        "ProfitLossPercent": (half_kelly_profit / bankroll_value) * 100.0 if bankroll_value > 0 else 0.0, # Half-Kelly profit or loss percentage is stored
                        "BetsPlaced": half_kelly_bets_placed, # Number of fixtures where Half-Kelly placed a bet is stored
                        "TotalStaked": half_kelly_total_staked, # Total Half-Kelly staked amount is stored
                        "ROI_On_Stake": (half_kelly_profit / half_kelly_total_staked) if half_kelly_total_staked > 0 else 0.0, # Half-Kelly ROI on staked amount is stored
                        "Return_On_Bankroll": (half_kelly_profit / bankroll_value) if bankroll_value > 0 else 0.0, # Half-Kelly return on starting bankroll is stored
                        "MaxDropFromPeak": float(betting_log["HalfKellyDropFromPeak"].max()) if not betting_log.empty else 0.0, # Maximum Half-Kelly drop from peak is stored
                        "StoppedEarly": half_kelly_stopped_early, # Flag indicating whether Half-Kelly stopped early is stored
                        "StopDate": half_kelly_stop_date, # The date of the fixture when Half-Kelly stopped early is stored
                    },
                    { # Intelligent Kelly summary row is added
                        "Model": model_name, # Model name is stored in summary row
                        "Strategy": "Intelligent Kelly", # Strategy label for summary row is stored
                        "InitialBankroll": bankroll_value, # Initial bankroll used for strategy is stored
                        "FinalBalance": float(intelligent_kelly_balance), # Final Intelligent Kelly bankroll is stored
                        "Profit": intelligent_kelly_profit, # Intelligent Kelly profit is stored
                        "ProfitLossPercent": (intelligent_kelly_profit / bankroll_value) * 100.0 if bankroll_value > 0 else 0.0, # Intelligent Kelly profit or loss percentage is stored
                        "BetsPlaced": intelligent_kelly_bets_placed, # Number of fixtures where Intelligent Kelly placed a bet is stored
                        "TotalStaked": intelligent_kelly_total_staked, # Total Intelligent Kelly staked amount is stored
                        "ROI_On_Stake": (intelligent_kelly_profit / intelligent_kelly_total_staked) if intelligent_kelly_total_staked > 0 else 0.0, # Intelligent Kelly ROI on staked amount is stored
                        "Return_On_Bankroll": (intelligent_kelly_profit / bankroll_value) if bankroll_value > 0 else 0.0, # Intelligent Kelly return on starting bankroll is stored
                        "MaxDropFromPeak": float(betting_log["IntelligentKellyDropFromPeak"].max()) if not betting_log.empty else 0.0, # Maximum Intelligent Kelly drop from peak is stored
                        "StoppedEarly": intelligent_kelly_stopped_early, # Flag indicating whether Intelligent Kelly stopped early is stored
                        "StopDate": intelligent_kelly_stop_date, # The date of the fixture when Intelligent Kelly stopped early is stored
                    },
                    { # Intelligent Half-Kelly summary row is added
                        "Model": model_name, # Model name is stored in summary row
                        "Strategy": "Intelligent Half-Kelly", # Strategy label for summary row is stored
                        "InitialBankroll": bankroll_value, # Initial bankroll used for strategy is stored
                        "FinalBalance": float(intelligent_half_kelly_balance), # Final Intelligent Half-Kelly bankroll is stored
                        "Profit": intelligent_half_kelly_profit, # Intelligent Half-Kelly profit is stored
                        "ProfitLossPercent": (intelligent_half_kelly_profit / bankroll_value) * 100.0 if bankroll_value > 0 else 0.0, # Intelligent Half-Kelly profit or loss percentage is stored
                        "BetsPlaced": intelligent_half_kelly_bets_placed, # Number of fixtures where Intelligent Half-Kelly placed a bet is stored
                        "TotalStaked": intelligent_half_kelly_total_staked, # Total Intelligent Half-Kelly staked amount is stored
                        "ROI_On_Stake": (intelligent_half_kelly_profit / intelligent_half_kelly_total_staked) if intelligent_half_kelly_total_staked > 0 else 0.0, # Intelligent Half-Kelly ROI on staked amount is stored
                        "Return_On_Bankroll": (intelligent_half_kelly_profit / bankroll_value) if bankroll_value > 0 else 0.0, # Intelligent Half-Kelly return on starting bankroll is stored
                        "MaxDropFromPeak": float(betting_log["IntelligentHalfKellyDropFromPeak"].max()) if not betting_log.empty else 0.0, # Maximum Intelligent Half-Kelly drop from peak is stored
                        "StoppedEarly": intelligent_half_kelly_stopped_early, # Flag indicating whether Intelligent Half-Kelly stopped early is stored
                        "StopDate": intelligent_half_kelly_stop_date, # The date of the fixture when Intelligent Half-Kelly stopped early is stored
                    },
                    { # Flat prediction summary row is added
                        "Model": model_name, # Model name is stored in summary row
                        "Strategy": "Flat Prediction", # Strategy label for summary row is stored
                        "InitialBankroll": bankroll_value, # Initial bankroll used for strategy is stored
                        "FinalBalance": float(flat_prediction_balance), # Final flat prediction bankroll is stored
                        "Profit": flat_prediction_profit, # Flat prediction profit is stored
                        "ProfitLossPercent": (flat_prediction_profit / bankroll_value) * 100.0 if bankroll_value > 0 else 0.0, # Flat prediction profit or loss percentage is stored
                        "BetsPlaced": flat_prediction_bets_placed, # Number of fixtures where the flat prediction strategy placed a bet is stored
                        "TotalStaked": flat_prediction_total_staked, # Total flat prediction staked amount is stored
                        "ROI_On_Stake": (flat_prediction_profit / flat_prediction_total_staked) if flat_prediction_total_staked > 0 else 0.0, # Flat prediction ROI on staked amount is stored
                        "Return_On_Bankroll": (flat_prediction_profit / bankroll_value) if bankroll_value > 0 else 0.0, # Flat prediction return on starting bankroll is stored
                        "MaxDropFromPeak": float(betting_log["FlatPredictionDropFromPeak"].max()) if not betting_log.empty else 0.0, # Maximum flat prediction drop from peak is stored
                        "StoppedEarly": flat_prediction_stopped_early, # Flag indicating whether the flat prediction strategy stopped early is stored
                        "StopDate": flat_prediction_stop_date, # The date of the fixture when the flat prediction strategy stopped early is stored
                    },
                ])

                if bankroll_value == standard_bankroll_value: # If this is the bankroll value used for the main simulation
                    ensure_dirs(str(prediction_file.parent)) # Ensures the model result directory exists
                    betting_log_path = prediction_file.parent / f"{model_name}_{prediction_year}_kelly_half_kelly_betting_log.csv" # Betting log output path is built
                    summary_path = prediction_file.parent / f"{model_name}_{prediction_year}_kelly_half_kelly_summary.csv" # Model summary output path is built
                    betting_log.to_csv(betting_log_path, index=False) # Match by match breakdown csv is saved
                    summary.to_csv(summary_path, index=False) # Summary csv is saved
                    print(f"Saved Kelly/Half-Kelly betting log: {betting_log_path}") # Confirmation message for betting log is printed
                    print(f"Saved Kelly/Half-Kelly summary: {summary_path}") # Confirmation message for summary is printed
                    combined_summaries.append(summary) # Per model summary dataframe is added to combined summary list

                if bankroll_value in starting_bankroll_values: # If this bankroll value is not used for the main simulation
                    starting_bankroll_summary_rows.extend(summary[["Model", "Strategy", "InitialBankroll", "FinalBalance", "Profit", "ProfitLossPercent"]].to_dict("records")) # The summary columns required are selected from the summary dataframe with each row being a dictionary

        combined_summary = pd.concat(combined_summaries, ignore_index=True) # All model summaries are concatenated into one dataframe
        ensure_dirs(str(results_path / "betting_simulation" / str(prediction_year))) # Ensures the year specific betting simulation results directory exists
        combined_summary_path = results_path / "betting_simulation" / str(prediction_year) / "kelly_half_kelly_model_comparison.csv" # Combined summary output path is built
        combined_summary.to_csv(combined_summary_path, index=False) # Combined models summary csv is saved
        print(f"Saved models Kelly/Half-Kelly comparison: {combined_summary_path}") # Confirmation message for combined output is printed

        starting_bankroll_summary_rows_df = pd.DataFrame(starting_bankroll_summary_rows).sort_values(["Model", "Strategy", "InitialBankroll"]).reset_index(drop=True) # The starting bankroll summary rows list is converted into a dataframe sorting the rows by Model, Strategy and InitialBankroll resetting the index at the end
        bankroll_profit_csv_path = results_path / "betting_simulation" / str(prediction_year) / "kelly_half_kelly_starting_bankroll_vs_final_profit.csv" # Bankroll comparison csv output path is built
        starting_bankroll_summary_rows_df.to_csv(bankroll_profit_csv_path, index=False) # Bankroll comparison csv is saved
        print(f"Saved Kelly/Half-Kelly starting bankroll vs final profit csv: {bankroll_profit_csv_path}") # Confirmation message for the bankroll comparison csv is printed

        bankroll_plot_path = results_path / "betting_simulation" / str(prediction_year) / "kelly_half_kelly_starting_bankroll_vs_final_profit.png" # Bankroll comparison chart output path is built
        strategy_order = ["Flat Prediction", "Kelly", "Half-Kelly", "Intelligent Kelly", "Intelligent Half-Kelly"] # The strategy order for plotting is defined
        colour_map = plt.get_cmap("tab20") # A categorical colour map is called to assign different colours to the lines
        fig, ax = plt.subplots(figsize=(16, 9)) # A figure and axis are created for the starting bankroll plot
        line_index = 0 # A counter representing the line index which is used to give each line a different colour is initialised
        for model_name in starting_bankroll_summary_rows_df["Model"].drop_duplicates(): # For each unique model in the bankroll comparison dataframe
            for strategy_name in strategy_order: # For each betting strategy
                line_df = starting_bankroll_summary_rows_df[(starting_bankroll_summary_rows_df["Model"] == model_name) & (starting_bankroll_summary_rows_df["Strategy"] == strategy_name)] # The dataframe is filtered for rows of the current model and current strategy
                if line_df.empty: # If the filtered dataframe returned no rows
                    continue # This line is not plotted
                ax.plot(line_df["InitialBankroll"], line_df["Profit"], marker="o", linewidth=2, color=colour_map(line_index), label=f"{model_name} - {strategy_name}") # A line for the respective model and strategy with InitialBankroll on the x-axis, Profit on the y-axis, a marker on each point, a line thickness of 2, a unique colour and a legend label is plotted
                line_index += 1 # The line index counter is incremented so the next line gets a different colour
        ax.set_xlabel("Starting Bankroll") # The x-axis label is set
        ax.set_ylabel("Final Profit at the end of the season") # The y-axis label is set
        ax.set_xticks(starting_bankroll_values) # The tested bankroll values are marked on the x-axis
        ax.grid(True, alpha=0.3) # Grid lines of transparency 0.3 are added across the y-axis
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9) # The legend is added and is positioned outside the plot on the right
        fig.tight_layout() # Layout is adjusted
        fig.savefig(bankroll_plot_path, dpi=200, bbox_inches="tight") # Image is saved to the specified path setting the resolution and cropping around the table removing extra whitespace
        plt.close(fig) # Figure is closed
        print(f"Saved Kelly/Half-Kelly starting bankroll vs final profit plot: {bankroll_plot_path}") # Confirmation message for the bankroll comparison chart is printed

if __name__ == "__main__":
    betting_simulation()