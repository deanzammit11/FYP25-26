from pathlib import Path
import pandas as pd
from src.utils import ensure_dirs, extract_model_name_from_filename

def betting_simulation(initial_bankroll=100.0, max_fraction=1.0, min_edge=0.0):
    initial_bankroll = float(initial_bankroll) # Starting bankroll is converted to float
    max_fraction = float(max_fraction) # Maximum allowed stake fraction is converted to float
    min_edge = float(min_edge) # Minimum Kelly edge required to place a bet is converted to float
    fixture_columns = ["Season", "Date", "HomeTeam", "AwayTeam"] # Fixture columns are defined
    probability_columns = ["ProbHome", "ProbDraw", "ProbAway"] # Model probability columns are defined
    odds_columns = ["Bet365HomeWinOdds", "Bet365DrawOdds", "Bet365AwayWinOdds"] # Odds columns are defined
    intelligent_columns = ["EloTierHome", "EloTierAway"] # Elo tier columns are defined

    results_path = Path("data/results") # Results directory path is defined
    prediction_files = sorted(results_path.glob("*/*_2023_predictions.csv")) # The model prediction files for 2023 are found and sorted
    if not prediction_files: # If no prediction files were found
        raise FileNotFoundError(f"No prediction files found under {results_path}") # A file not found error is raised

    combined_summaries = [] # A list to store each model summary dataframe is initialised
    training_data_path = Path("data/features/eng1_data_combined.csv") # Source path for training data is defined
    if not training_data_path.exists(): # If training data file was not found
        raise FileNotFoundError(f"Training data file not found: {training_data_path}") # A file not found error is raised
    training_data_df = pd.read_csv(training_data_path) # Training data dataframe is loaded
    required_training_columns = {"Season", "ResultEncoded", "EloTierHome", "EloTierAway"} # Required columns to build Elo tier matchup history are defined
    missing_training_columns = sorted(required_training_columns.difference(training_data_df.columns)) # Any missing required training columns are identified
    if missing_training_columns: # If any required columns are missing
        raise ValueError(f"Missing required columns in {training_data_path}: {missing_training_columns}") # A value error is raised
    training_data_df["Season"] = pd.to_numeric(training_data_df["Season"], errors="coerce") # Season column in training data is converted to numeric
    training_base = training_data_df[training_data_df["Season"] < 2023].copy() # Only training seasons are kept for initial matchup history
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

        kelly_balance = initial_bankroll # Kelly bankroll is initialised with the starting bankroll
        half_kelly_balance = initial_bankroll # Half-Kelly bankroll is initialised with the starting bankroll
        intelligent_kelly_balance = initial_bankroll # Intelligent Kelly bankroll is initialised with the starting bankroll
        intelligent_half_kelly_balance = initial_bankroll # Intelligent Half-Kelly bankroll is initialised with the starting bankroll
        peak_kelly = initial_bankroll # Peak Kelly bankroll value is initialised
        peak_half_kelly = initial_bankroll # Peak Half-Kelly bankroll value is initialised
        peak_intelligent_kelly = initial_bankroll # Peak Intelligent Kelly bankroll value is initialised
        peak_intelligent_half_kelly = initial_bankroll # Peak Intelligent Half-Kelly bankroll value is initialised
        rows = [] # An empty list which will store a dictionary for each fixture containing the data related to the bet on that fixture is intialised
        matchup_totals = {} # An empty dictionary which will store the total number of historical matches for each Elo tier home vs away pairing is initialised
        matchup_outcome_counts = {} # An empty dictionary which will store the total historical outcome counts for each Elo tier home vs away pairing is initialised

        for _, row in training_base.iterrows(): # For each training set fixture
            home_tier = int(row["EloTierHome"]) # Home Elo tier is converted to integer
            away_tier = int(row["EloTierAway"]) # Away Elo tier is converted to integer
            outcome = int(row["ResultEncoded"]) # Match outcome is converted to integer
            tier_key = (home_tier, away_tier) # Elo tier matchup key is created and stored
            outcome_key = (home_tier, away_tier, outcome) # Elo tier matchup and outcome key is created and stored
            matchup_totals[tier_key] = matchup_totals.get(tier_key, 0) + 1 # The elo tier key is looked up and if it is found it is returned and incremented and if it is not found meaning it is the first appearance it is intialised to 0 and incremented
            matchup_outcome_counts[outcome_key] = matchup_outcome_counts.get(outcome_key, 0) + 1 # The elo tier and outcome key is looked up and if it is found it is returned and incremented and if it is not found meaning it is the first appearance it is intialised to 0 and incremented

        for _, row in df.iterrows(): # For each predicted fixture in chronological order
            options = [ # A list which each iteration storing a tuple for each outcome along with the outcome label, probability and odds for that fixture is built
                ("Home", 1, row["ProbHome"], row["Bet365HomeWinOdds"]), # Home option tuple
                ("Draw", 0, row["ProbDraw"], row["Bet365DrawOdds"]), # Draw option tuple
                ("Away", -1, row["ProbAway"], row["Bet365AwayWinOdds"]), # Away option tuple
            ]

            best = None # Best bet candidate for the current fixture is initialised as None
            for label, encoded, probability, odds in options: # For each possible outcome option
                probability = float(probability) # Option probability is converted to float
                q = 1.0 - probability # Option loss probability is calculated
                b = float(odds) - 1.0 # Net odds are calculated from odds
                if b <= 0: # If net odds are invalid
                    continue # Skip this option
                edge = probability - (q / b) # Kelly edge is calculated
                if edge <= min_edge: # If the edge does not exceed minimum edge needed to place a bet
                    continue # Skip this option
                kelly_fraction = max(0.0, probability - (q / b)) # Kelly fraction is computed and if it is negative it is set to 0
                bet_option = { # Bet dictionary for respective option is built
                    "BetOn": label, # Bet label is stored
                    "BetOnEncoded": encoded, # Encoded bet result is stored
                    "p": probability, # Option win probability is stored
                    "q": q, # Option loss probability is stored
                    "b": b, # Net odds are stored
                    "Odds": float(odds), # Odds are stored
                    "KellyFraction": kelly_fraction, # Kelly fraction is stored
                }
                if best is None or bet_option["KellyFraction"] > best["KellyFraction"]: # If first bet option or better Kelly fraction than current best option
                    best = bet_option # Current bet option becomes best bet option

            kelly_before = kelly_balance # Kelly bankroll before fixture is stored
            half_kelly_before = half_kelly_balance # Half-Kelly bankroll before fixture is stored
            intelligent_kelly_before = intelligent_kelly_balance # Intelligent Kelly bankroll before fixture is stored
            intelligent_half_kelly_before = intelligent_half_kelly_balance # Intelligent Half-Kelly bankroll before fixture is stored
            home_tier_value = int(row["EloTierHome"]) if pd.notna(row["EloTierHome"]) else pd.NA # Current fixture home Elo tier is captured and if not found it is set to NA
            away_tier_value = int(row["EloTierAway"]) if pd.notna(row["EloTierAway"]) else pd.NA # Current fixture away Elo tier is captured and if not found it is set to NA

            base_fixture_row = { # Base fixture fields present in each row are stored in a dictionary
                "Model": model_name, # Model name is stored
                "Date": row["Date"], # Fixture date is stored
                "HomeTeam": row["HomeTeam"], # Home team is stored
                "AwayTeam": row["AwayTeam"], # Away team is stored
                "BetPlaced": bool(best is not None), # Flag indicating whether bet is placed is stored
                "ActualResult": int(row["ResultEncoded"]), # Actual outcome is stored
                "EloTierHome": home_tier_value, # Home Elo tier is stored
                "EloTierAway": away_tier_value, # Away Elo tier is stored
            }

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
                    "KellyBalanceBefore": kelly_before, # Kelly bankroll before fixture is stored
                    "KellyBalanceAfter": kelly_balance, # Kelly bankroll after fixture is stored
                    "HalfKellyBalanceBefore": half_kelly_before, # Half-Kelly bankroll before fixture is stored
                    "HalfKellyBalanceAfter": half_kelly_balance, # Half-Kelly bankroll after fixture is stored
                    "IntelligentKellyBalanceBefore": intelligent_kelly_before, # Intelligent Kelly bankroll before fixture is stored
                    "IntelligentKellyBalanceAfter": intelligent_kelly_balance, # Intelligent Kelly bankroll after fixture is stored
                    "IntelligentHalfKellyBalanceBefore": intelligent_half_kelly_before, # Intelligent Half-Kelly bankroll before fixture is stored
                    "IntelligentHalfKellyBalanceAfter": intelligent_half_kelly_balance, # Intelligent Half-Kelly bankroll after fixture is stored
                    "KellyDropFromPeak": (peak_kelly - kelly_balance) / peak_kelly if peak_kelly > 0 else 0.0, # Kelly drop from peak after fixture is stored
                    "HalfKellyDropFromPeak": (peak_half_kelly - half_kelly_balance) / peak_half_kelly if peak_half_kelly > 0 else 0.0, # Half-Kelly drop from peak after fixture is stored
                    "IntelligentKellyDropFromPeak": (peak_intelligent_kelly - intelligent_kelly_balance) / peak_intelligent_kelly if peak_intelligent_kelly > 0 else 0.0, # Intelligent Kelly drop from peak after fixture is stored
                    "IntelligentHalfKellyDropFromPeak": (peak_intelligent_half_kelly - intelligent_half_kelly_balance) / peak_intelligent_half_kelly if peak_intelligent_half_kelly > 0 else 0.0, # Intelligent Half-Kelly drop from peak after fixture is stored
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
                    raise ValueError(f"Insufficient Elo tier matchup history for fixture: {row['Date']} {row['HomeTeam']} vs {row['AwayTeam']} " f"(home tier={home_tier_value}, away tier={away_tier_value})") # A value error is raised
                intelligent_multiplier = matchup_outcome_counts.get(outcome_key, 0) / historical_total # Intelligent multiplier is calculated

                intelligent_kelly_fraction = min(max_fraction, max(0.0, kelly_fraction * intelligent_multiplier)) # Intelligent Kelly fraction is set where if it is negative it is set to 0 and if it is larger than the max fraction the max fraction is taken
                intelligent_half_kelly_fraction = min(max_fraction, max(0.0, half_kelly_fraction * intelligent_multiplier)) # Intelligent Half-Kelly fraction is set where if it is negative it is set to 0 and if it is larger than the max fraction the max fraction is taken

                kelly_stake = kelly_before * kelly_fraction # Kelly stake is calculated
                half_kelly_stake = half_kelly_before * half_kelly_fraction # Half-Kelly stake is calculated
                intelligent_kelly_stake = intelligent_kelly_before * intelligent_kelly_fraction # Intelligent Kelly stake is calculated
                intelligent_half_kelly_stake = intelligent_half_kelly_before * intelligent_half_kelly_fraction # Intelligent Half-Kelly stake is calculated

                won = int(row["ResultEncoded"]) == int(best["BetOnEncoded"]) # It is checked if bet won
                kelly_profit_loss = kelly_stake * best["b"] if won else -kelly_stake # Kelly profit or loss is calculated
                half_kelly_profit_loss = half_kelly_stake * best["b"] if won else -half_kelly_stake # Half-Kelly profit or loss is calculated
                intelligent_kelly_profit_loss = intelligent_kelly_stake * best["b"] if won else -intelligent_kelly_stake # Intelligent Kelly profit or loss is calculated
                intelligent_half_kelly_profit_loss = intelligent_half_kelly_stake * best["b"] if won else -intelligent_half_kelly_stake # Intelligent Half-Kelly profit or loss is calculated

                kelly_balance = kelly_before + kelly_profit_loss # Kelly bankroll is updated
                half_kelly_balance = half_kelly_before + half_kelly_profit_loss # Half-Kelly bankroll is updated
                intelligent_kelly_balance = intelligent_kelly_before + intelligent_kelly_profit_loss # Intelligent Kelly bankroll is updated
                intelligent_half_kelly_balance = intelligent_half_kelly_before + intelligent_half_kelly_profit_loss # Intelligent Half-Kelly bankroll is updated
                peak_kelly = max(peak_kelly, kelly_balance) # Peak Kelly bankroll is updated
                peak_half_kelly = max(peak_half_kelly, half_kelly_balance) # Peak Half-Kelly bankroll is updated
                peak_intelligent_kelly = max(peak_intelligent_kelly, intelligent_kelly_balance) # Peak Intelligent Kelly bankroll is updated
                peak_intelligent_half_kelly = max(peak_intelligent_half_kelly, intelligent_half_kelly_balance) # Peak Intelligent Half-Kelly bankroll is updated

                rows.append({ # Bet fixture row is appended
                    **base_fixture_row, # The base fixture fields are stored
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
                    "KellyBalanceBefore": kelly_before, # Kelly bankroll before fixture is stored
                    "KellyBalanceAfter": kelly_balance, # Kelly bankroll after fixture is stored
                    "HalfKellyBalanceBefore": half_kelly_before, # Half-Kelly bankroll before fixture is stored
                    "HalfKellyBalanceAfter": half_kelly_balance, # Half-Kelly bankroll after fixture is stored
                    "IntelligentKellyBalanceBefore": intelligent_kelly_before, # Intelligent Kelly bankroll before fixture is stored
                    "IntelligentKellyBalanceAfter": intelligent_kelly_balance, # Intelligent Kelly bankroll after fixture is stored
                    "IntelligentHalfKellyBalanceBefore": intelligent_half_kelly_before, # Intelligent Half-Kelly bankroll before fixture is stored
                    "IntelligentHalfKellyBalanceAfter": intelligent_half_kelly_balance, # Intelligent Half-Kelly bankroll after fixture is stored
                    "KellyDropFromPeak": (peak_kelly - kelly_balance) / peak_kelly if peak_kelly > 0 else 0.0, # Kelly drop from peak after fixture is stored
                    "HalfKellyDropFromPeak": (peak_half_kelly - half_kelly_balance) / peak_half_kelly if peak_half_kelly > 0 else 0.0, # Half-Kelly drop from peak after fixture is stored
                    "IntelligentKellyDropFromPeak": (peak_intelligent_kelly - intelligent_kelly_balance) / peak_intelligent_kelly if peak_intelligent_kelly > 0 else 0.0, # Intelligent Kelly drop from peak after fixture is stored
                    "IntelligentHalfKellyDropFromPeak": (peak_intelligent_half_kelly - intelligent_half_kelly_balance) / peak_intelligent_half_kelly if peak_intelligent_half_kelly > 0 else 0.0, # Intelligent Half-Kelly drop from peak after fixture is stored
                })

            if pd.notna(home_tier_value) and pd.notna(away_tier_value) and pd.notna(row["ResultEncoded"]): # If current fixture Elo tiers and actual outcome are available
                tier_key = (int(home_tier_value), int(away_tier_value)) # Elo tier matchup key is created
                outcome_key = (int(home_tier_value), int(away_tier_value), int(row["ResultEncoded"])) # Elo tier matchup and actual outcome key is created and stored
                matchup_totals[tier_key] = matchup_totals.get(tier_key, 0) + 1 # Elo tier matchup pair total count is incremented
                matchup_outcome_counts[outcome_key] = matchup_outcome_counts.get(outcome_key, 0) + 1 # Elo tier matchup pair outcome count is incremented

        betting_log = pd.DataFrame(rows) # Fixture rows are converted into a dataframe
        placed_bets = betting_log[betting_log["BetPlaced"]] # Rows where a bet was placed are filtered

        kelly_profit = float(kelly_balance - initial_bankroll) # Total Kelly profit is calculated
        half_kelly_profit = float(half_kelly_balance - initial_bankroll) # Total Half-Kelly profit is calculated
        intelligent_kelly_profit = float(intelligent_kelly_balance - initial_bankroll) # Total Intelligent Kelly profit is calculated
        intelligent_half_kelly_profit = float(intelligent_half_kelly_balance - initial_bankroll) # Total Intelligent Half-Kelly profit is calculated
        kelly_total_staked = float(betting_log["KellyStake"].sum()) # Total Kelly staked amount is calculated
        half_kelly_total_staked = float(betting_log["HalfKellyStake"].sum()) # Total Half-Kelly staked amount is calculated
        intelligent_kelly_total_staked = float(betting_log["IntelligentKellyStake"].sum()) # Total Intelligent Kelly staked amount is calculated
        intelligent_half_kelly_total_staked = float(betting_log["IntelligentHalfKellyStake"].sum()) # Total Intelligent Half-Kelly staked amount is calculated

        summary = pd.DataFrame([ # A summary dataframe is built
            { # Kelly summary row is added
                "Model": model_name, # Model name is stored in summary row
                "Strategy": "Kelly", # Strategy label for summary row is stored
                "InitialBankroll": initial_bankroll, # Initial bankroll used for strategy is stored
                "FinalBalance": float(kelly_balance), # Final Kelly bankroll is stored
                "Profit": kelly_profit, # Kelly profit is stored
                "BetsPlaced": int(len(placed_bets)), # Number of placed bets is stored
                "TotalStaked": kelly_total_staked, # Total Kelly staked amount is stored
                "ROI_On_Stake": (kelly_profit / kelly_total_staked) if kelly_total_staked > 0 else 0.0, # Kelly ROI on staked amount is stored
                "Return_On_Bankroll": (kelly_profit / initial_bankroll) if initial_bankroll > 0 else 0.0, # Kelly return on starting bankroll is stored
                "MaxDropFromPeak": float(betting_log["KellyDropFromPeak"].max()) if not betting_log.empty else 0.0, # Maximum Kelly drop from peak is stored
            },
            { # Half-Kelly summary row is added
                "Model": model_name, # Model name is stored in summary row
                "Strategy": "Half-Kelly", # Strategy label for summary row is stored
                "InitialBankroll": initial_bankroll, # Initial bankroll used for strategy is stored
                "FinalBalance": float(half_kelly_balance), # Final Half-Kelly bankroll is stored
                "Profit": half_kelly_profit, # Half-Kelly profit is stored
                "BetsPlaced": int(len(placed_bets)), # Number of placed bets is stored
                "TotalStaked": half_kelly_total_staked, # Total Half-Kelly staked amount is stored
                "ROI_On_Stake": (half_kelly_profit / half_kelly_total_staked) if half_kelly_total_staked > 0 else 0.0, # Half-Kelly ROI on staked amount is stored
                "Return_On_Bankroll": (half_kelly_profit / initial_bankroll) if initial_bankroll > 0 else 0.0, # Half-Kelly return on starting bankroll is stored
                "MaxDropFromPeak": float(betting_log["HalfKellyDropFromPeak"].max()) if not betting_log.empty else 0.0, # Maximum Half-Kelly drop from peak is stored
            },
            { # Intelligent Kelly summary row is added
                "Model": model_name, # Model name is stored in summary row
                "Strategy": "Intelligent Kelly", # Strategy label for summary row is stored
                "InitialBankroll": initial_bankroll, # Initial bankroll used for strategy is stored
                "FinalBalance": float(intelligent_kelly_balance), # Final Intelligent Kelly bankroll is stored
                "Profit": intelligent_kelly_profit, # Intelligent Kelly profit is stored
                "BetsPlaced": int(len(placed_bets)), # Number of placed bets is stored
                "TotalStaked": intelligent_kelly_total_staked, # Total Intelligent Kelly staked amount is stored
                "ROI_On_Stake": (intelligent_kelly_profit / intelligent_kelly_total_staked) if intelligent_kelly_total_staked > 0 else 0.0, # Intelligent Kelly ROI on staked amount is stored
                "Return_On_Bankroll": (intelligent_kelly_profit / initial_bankroll) if initial_bankroll > 0 else 0.0, # Intelligent Kelly return on starting bankroll is stored
                "MaxDropFromPeak": float(betting_log["IntelligentKellyDropFromPeak"].max()) if not betting_log.empty else 0.0, # Maximum Intelligent Kelly drop from peak is stored
            },
            { # Intelligent Half-Kelly summary row is added
                "Model": model_name, # Model name is stored in summary row
                "Strategy": "Intelligent Half-Kelly", # Strategy label for summary row is stored
                "InitialBankroll": initial_bankroll, # Initial bankroll used for strategy is stored
                "FinalBalance": float(intelligent_half_kelly_balance), # Final Intelligent Half-Kelly bankroll is stored
                "Profit": intelligent_half_kelly_profit, # Intelligent Half-Kelly profit is stored
                "BetsPlaced": int(len(placed_bets)), # Number of placed bets is stored
                "TotalStaked": intelligent_half_kelly_total_staked, # Total Intelligent Half-Kelly staked amount is stored
                "ROI_On_Stake": (intelligent_half_kelly_profit / intelligent_half_kelly_total_staked) if intelligent_half_kelly_total_staked > 0 else 0.0, # Intelligent Half-Kelly ROI on staked amount is stored
                "Return_On_Bankroll": (intelligent_half_kelly_profit / initial_bankroll) if initial_bankroll > 0 else 0.0, # Intelligent Half-Kelly return on starting bankroll is stored
                "MaxDropFromPeak": float(betting_log["IntelligentHalfKellyDropFromPeak"].max()) if not betting_log.empty else 0.0, # Maximum Intelligent Half-Kelly drop from peak is stored
            },
        ])

        ensure_dirs(str(prediction_file.parent)) # Ensures the model result directory exists
        betting_log_path = prediction_file.parent / f"{model_name}_2023_kelly_half_kelly_betting_log.csv" # Betting log output path is built
        summary_path = prediction_file.parent / f"{model_name}_2023_kelly_half_kelly_summary.csv" # Model summary output path is built
        betting_log.to_csv(betting_log_path, index=False) # Match by match breakdown csv is saved
        summary.to_csv(summary_path, index=False) # Summary csv is saved
        print(f"Saved Kelly/Half-Kelly betting log: {betting_log_path}") # Confirmation message for betting log is printed
        print(f"Saved Kelly/Half-Kelly summary: {summary_path}") # Confirmation message for summary is printed

        combined_summaries.append(summary) # Per model summary dataframe is added to combined summary list

    combined_summary = pd.concat(combined_summaries, ignore_index=True) # All model summaries are concatenated into one dataframe
    ensure_dirs(str(results_path)) # Ensures results directory exists
    combined_summary_path = results_path / "kelly_half_kelly_model_comparison.csv" # Combined summary output path is built
    combined_summary.to_csv(combined_summary_path, index=False) # Combined models summary csv is saved
    print(f"Saved models Kelly/Half-Kelly comparison: {combined_summary_path}") # Confirmation message for combined output is printed

if __name__ == "__main__":
    betting_simulation() # Betting simulation is run using default parameter values when this module is executed directly