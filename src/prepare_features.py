import pandas as pd
from src.utils import ensure_dirs, save_csv

def prepare_features():
    ensure_dirs("data/features") # Checks if directory exists and if it does not it creates it
    df = pd.read_csv("data/processed/eng1_all_seasons.csv") # Csv file is read from the specified directory and stored in a data frame

    df = df.dropna(subset=["FullTimeResult"]) # Drop rows with missing results

    df["ResultEncoded"] = df["FullTimeResult"].map({"H": 1, "D": 0, "A": -1}) # Encode target
    df["HomePoints"] = df["ResultEncoded"].map({1: 3, 0: 1, -1: 0}) # Result is mapped to points
    df["AwayPoints"] = df["ResultEncoded"].map({1: 0, 0: 1, -1: 3}) # Result is mapped to points
    
    # Betting Odds Features
    # Bet365 odds are converted into implied probabilities
    df["Home"] = 1 / df["Bet365HomeWinOdds"]
    df["Draw"] = 1 / df["Bet365DrawOdds"]
    df["Away"] = 1 / df["Bet365AwayWinOdds"]

    df["Total"] = df["Home"] + df["Draw"] + df["Away"] # Calculate the total of the implied probabilities

    df["Bet365HomeWinOddsPercentage"] = df["Home"] / df["Total"] # Convert the implied home win probability into a percentage
    df["Bet365DrawOddsPercentage"] = df["Draw"] / df["Total"] # Convert the implied draw probability into a percentage
    df["Bet365AwayWinOddsPercentage"] = df["Away"] / df["Total"] # Convert the implied away win probability into a percentage

    df["OddsDifference_HvA"] = (df["Bet365HomeWinOddsPercentage"] - df["Bet365AwayWinOddsPercentage"]) # Home vs away odds difference without absolute value
    df["OddsDifference_HvD"] = (df["Bet365HomeWinOddsPercentage"] - df["Bet365DrawOddsPercentage"]) # Home vs draw odds difference without absolute value
    df["OddsDifference_AvD"] = (df["Bet365AwayWinOddsPercentage"] - df["Bet365DrawOddsPercentage"]) # Away vs draw odds difference without absolute value
    # df["OddsDifference_HvA"] = (df["Bet365HomeWinOddsPercentage"] - df["Bet365AwayWinOddsPercentage"]).abs() # Home vs away odds difference with absolute value
    # df["OddsDifference_HvD"] = (df["Bet365HomeWinOddsPercentage"] - df["Bet365DrawOddsPercentage"]).abs() # Home vs draw odds difference with absolute value
    # df["OddsDifference_AvD"] = (df["Bet365AwayWinOddsPercentage"] - df["Bet365DrawOddsPercentage"]).abs() # Away vs draw odds difference with absolute value

    odds_cols = [ # Columns to store in odds csv file are selected
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "Bet365HomeWinOddsPercentage", "Bet365DrawOddsPercentage", "Bet365AwayWinOddsPercentage",
        "OddsDifference_HvA", "OddsDifference_HvD", "OddsDifference_AvD"
    ]
    df_odds = df[odds_cols].dropna() # Null values are removed
    save_csv(df_odds, "data/features/eng1_data_odds.csv") # Csv without null values is saved into the specified directory

    # Form Features
    df["HomeForm"] = (
        df["HomePoints"].groupby(df["HomeTeam"]).shift() # Each Home team's home results is shifted one game back
        .groupby(df["HomeTeam"]).rolling(window=5, min_periods=1).sum() # Sum of points scored by the home team over the last 5 home games is computed
        .reset_index(level=0, drop=True) # The HomeTeam grouping in the multi level index is dropped leaving only the row id
    )
    df["AwayForm"] = (
        df["AwayPoints"].groupby(df["AwayTeam"]).shift() # Each Away team's away results is shifted one game back
        .groupby(df["AwayTeam"]).rolling(window=5, min_periods=1).sum() # Sum of points scored by the away team over the last 5 away games is computed
        .reset_index(level=0, drop=True) # The AwayTeam grouping in the multi level index is dropped leaving only the row id
    )

    df["HomeAdvantageIndex"] = df["HomeForm"] - df["AwayForm"] # Calculates home advantage based on how the home side performs at home and how the away side performs away from home

    home_results = df[["Season", "Date", "HomeTeam", "HomePoints"]].rename(columns={"HomeTeam": "Team", "HomePoints": "Points"}) # A table storing the the points scored by the home team per fixture is created
    home_results["MatchIndex"] = df.index # Adds the row number to the home results
    away_results = df[["Season", "Date", "AwayTeam", "AwayPoints"]].rename(columns={"AwayTeam": "Team", "AwayPoints": "Points"}) # A table storing the the points scored by the away team per fixture is created
    away_results["MatchIndex"] = df.index # Adds the row number to the away results

    all_results = pd.concat([home_results, away_results], ignore_index=True) # Concatenates the home and away results into 1 results dataframe
    all_results = all_results.sort_values(["Season", "Team", "MatchIndex"]).drop(columns="MatchIndex") # Both tables are joined into one table and sorted by season, team name, and match order and the MatchIndex is then dropped

    grouped_points = all_results.groupby(["Season", "Team"])["Points"].shift() # Points scored are grouped by season and team and shifted one game back
    
    all_results["GeneralForm"] = (
        grouped_points.groupby([all_results["Season"], all_results["Team"]]) # Points are grouped again by Season and Team
        .rolling(window=5, min_periods=1).sum() # Sum of points scored over the last 5 games is computed
        .reset_index(level=[0, 1], drop=True) # The Season and Team grouping in the multi level index is dropped leaving only the row id
    )

    df = df.merge(
        all_results[["Season", "Team", "Date", "GeneralForm"]], # Merge df with the Season, Team, Date and GeneralForm columns in all_results
        how="left", # Keeps all columns in df and only merges the columns from all_results
        left_on=["Season", "HomeTeam", "Date"], right_on=["Season", "Team", "Date"] # Match Season, HomeTeam and Date from the left table and Season, Team and Date from the right table and if they match merge
    ).rename(columns={"GeneralForm": "HomeGeneralForm"}).drop(columns="Team") # GeneralForm column is renamed and the new Team column is removed

    df = df.merge(
        all_results[["Season", "Team", "Date", "GeneralForm"]], # Merge df with the Season, Team, Date and GeneralForm columns in all_results
        how="left", # Keeps all columns in df and only merges the columns from all_results
        left_on=["Season", "AwayTeam", "Date"], right_on=["Season", "Team", "Date"] # Match Season, AwayTeam and Date from the left table and Season, Team and Date from the right table and if they match merge
    ).rename(columns={"GeneralForm": "AwayGeneralForm"}).drop(columns="Team") # GeneralForm column is renamed and the new Team column is removed

    df["GeneralFormDifference"] = df["HomeGeneralForm"] - df["AwayGeneralForm"] # Points scored by the away team in the last 5 matches are subtracted from the points scored by the home team in the last 5 matches

    home_team_groups = df.groupby(["Season", "HomeTeam"]) # Rows are grouped by season and home team
    previous_home_scored = home_team_groups["FullTimeHomeGoals"].shift() # The values of the goals scored at home are shifted one row downwards
    df["AverageGoalsScoredAtHome"] = (
        previous_home_scored.groupby([df["Season"], df["HomeTeam"]]).cumsum() / home_team_groups.cumcount().replace(0, pd.NA) # The cumulative sum of the goals scored by the home team up to and excluding the current match is divded by the number of rows which have appeared in the group with zeros being replaced with NA since you cannot divide by 0
    ).fillna(0) # Null values are replaced with 0 where the first match is always 0

    previous_home_conceded = home_team_groups["FullTimeAwayGoals"].shift() # The values of the goals conceded at home are shifted one row downwards
    df["AverageGoalsConcededAtHome"] = (
        previous_home_conceded.groupby([df["Season"], df["HomeTeam"]]).cumsum() / home_team_groups.cumcount().replace(0, pd.NA) # The cumulative sum of the goals conceded by the home team up to and excluding the current match is divded by the number of rows which have appeared in the group with zeros being replaced with NA since you cannot divide by 0
    ).fillna(0) # Null values are replaced with 0 where the first match is always 0

    away_team_groups = df.groupby(["Season", "AwayTeam"]) # Rows are grouped by season and away team
    previous_away_scored = away_team_groups["FullTimeAwayGoals"].shift() # The values of the goals scored away from home are shifted one row downwards
    df["AverageGoalsScoredAtAway"] = (
        previous_away_scored.groupby([df["Season"], df["AwayTeam"]]).cumsum() / away_team_groups.cumcount().replace(0, pd.NA) # The cumulative sum of the goals scored by the away team away from home up to and excluding the current match is divded by the number of rows which have appeared in the group with zeros being replaced with NA since you cannot divide by 0
    ).fillna(0) # Null values are replaced with 0 where the first match is always 0

    previous_away_conceded = away_team_groups["FullTimeHomeGoals"].shift() # The values of the goals conceded away from home are shifted one row downwards
    df["AverageGoalsConcededAtAway"] = (
        previous_away_conceded.groupby([df["Season"], df["AwayTeam"]]).cumsum() / away_team_groups.cumcount().replace(0, pd.NA) # The cumulative sum of the goals conceded by the away team away from home up to and excluding the current match is divded by the number of rows which have appeared in the group with zeros being replaced with NA since you cannot divide by 0
    ).fillna(0)

    team_home_records = pd.DataFrame({ # A dataframe storing the season, date, team, goals scored and conceded and match index of the home team for each fixture is created
        "Season": df["Season"],
        "Date": df["Date"],
        "Team": df["HomeTeam"],
        "GoalsFor": df["FullTimeHomeGoals"],
        "GoalsAgainst": df["FullTimeAwayGoals"],
        "MatchIndex": df.index,
        "IsHome": True,
    })
    team_away_records = pd.DataFrame({ # A dataframe storing the season, date, team, goals scored and conceded and match index of the away team for each fixture is created
        "Season": df["Season"],
        "Date": df["Date"],
        "Team": df["AwayTeam"],
        "GoalsFor": df["FullTimeAwayGoals"],
        "GoalsAgainst": df["FullTimeHomeGoals"],
        "MatchIndex": df.index,
        "IsHome": False,
    })
    team_records = pd.concat([team_home_records, team_away_records], ignore_index=True) # Concatenates the home and away records into 1 records dataframe
    team_records = team_records.sort_values(["Season", "Team", "MatchIndex"]) # All records are sorted by Season, Team and MatchIndex
    team_groups = team_records.groupby(["Season", "Team"], sort=False) # All records are grouped by season and team
    team_records["TotalGoalsScored"] = team_groups["GoalsFor"].cumsum() - team_records["GoalsFor"] # The TotalGoalsScored excluding the current match are computed
    team_records["TotalGoalsConceded"] = team_groups["GoalsAgainst"].cumsum() - team_records["GoalsAgainst"] # The TotalGoalsConceded excluding the current match are computed
    home_totals = team_records[team_records["IsHome"]].set_index("MatchIndex") # The rows were IsHome is true are stored and their index is set to the original row number
    away_totals = team_records[~team_records["IsHome"]].set_index("MatchIndex") # The rows were IsHome is false are stored and their index is set to the original row number

    df["TotalGoalsScoredHome"] = home_totals["TotalGoalsScored"].reindex(df.index).fillna(0) # The TotalGoalsScored are stored in df while reindexing to contain the full index of df with instances where there were no prior values for that home fixture being filled with 0
    df["TotalGoalsConcededHome"] = home_totals["TotalGoalsConceded"].reindex(df.index).fillna(0) # The TotalGoalsConceded are stored in df while reindexing to contain the full index of df with instances where there were no prior values for that home fixture being filled with 0
    df["TotalGoalsScoredAway"] = away_totals["TotalGoalsScored"].reindex(df.index).fillna(0) # The TotalGoalsScored are stored in df while reindexing to contain the full index of df with instances where there were no prior values for that away fixture being filled with 0
    df["TotalGoalsConcededAway"] = away_totals["TotalGoalsConceded"].reindex(df.index).fillna(0) # The TotalGoalsConceded are stored in df while reindexing to contain the full index of df with instances where there were no prior values for that away fixture being filled with 0

    home_records = pd.DataFrame({ # A data frame consisting of 5 columns to store the records for the home side is built
        "Team": df["HomeTeam"], # Team name is the name of the home team
        "Opponent": df["AwayTeam"], # Opponent is the name of the away team
        "Points": df["HomePoints"], # Points is the number of points scored by the home team in that fixture
        "MatchIndex": df.index, # The respective row number in the original dataset
        "IsHome": True, # A boolean flag showing that the team played at home
    })
    away_records = pd.DataFrame({
        "Team": df["AwayTeam"], # Team name is the name of the away team
        "Opponent": df["HomeTeam"], # Opponent is the name of the home team
        "Points": df["AwayPoints"], # Points is the number of points scored by the away team in that fixture
        "MatchIndex": df.index, # The respective row number in the original dataset
        "IsHome": False, # A boolean flag showing that the team played away from home
    })

    head_to_head = pd.concat([home_records, away_records], ignore_index=True) # The home and away records are concatenated into a single dataframe with the index being reset.
    grouped_points = head_to_head.groupby(["Team", "Opponent"], sort=False)["Points"] # Each team, opponent pairing is grouped based on points while keeping the same order
    head_to_head["HistoricalPoints"] = (grouped_points.cumsum() - head_to_head["Points"]).fillna(0) # Computes the total points scored by a team against an opponent before the current match

    home_history = head_to_head[head_to_head["IsHome"]].set_index("MatchIndex")["HistoricalPoints"] # Selects the HistoricalPoints column for home records only and the index is replaced with the original index in df
    away_history = head_to_head[~head_to_head["IsHome"]].set_index("MatchIndex")["HistoricalPoints"] # Selects the HistoricalPoints column for away records only and the index is replaced with the original index in df

    df["HistoricalEncountersHome"] = home_history.reindex(df.index).fillna(0).astype(int) # Reindexes home_history to contain the full index of df with instances where there were no previous encounters for that home fixture being filled with 0
    df["HistoricalEncountersAway"] = away_history.reindex(df.index).fillna(0).astype(int) # Reindexes away_history to contain the full index of df with instances where there were no previous encounters for that away fixture being filled with 0

    form_cols = [ # Columns to store in form csv file are selected
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "HomeForm", "AwayForm", "HomeAdvantageIndex",
        "HomeGeneralForm", "AwayGeneralForm", "GeneralFormDifference",
        "AverageGoalsScoredAtHome", "AverageGoalsScoredAtAway",
        "AverageGoalsConcededAtHome", "AverageGoalsConcededAtAway",
        "TotalGoalsScoredHome", "TotalGoalsScoredAway",
        "TotalGoalsConcededHome", "TotalGoalsConcededAway",
        "HistoricalEncountersHome", "HistoricalEncountersAway"
    ]

    df_form = df[form_cols].fillna(0) # Sets missing values to 0 since form cannot be calculated on first appearance
    save_csv(df_form, "data/features/eng1_data_form.csv") # Csv without null values is saved into the specified directory

    # Ratings Features
    season_to_fifa_version = {2019: 20, 2020: 21, 2021: 22, 2022: 23, 2023: 24} # Maps the season to the respective fifa version
    season_team_name_map = { # Maps the team names from different datasets to each other by season
        2019: {
            "Liverpool": "Liverpool",
            "Man City": "Manchester City",
            "Man United": "Manchester United",
            "Chelsea": "Chelsea",
            "Leicester": "Leicester City",
            "Tottenham": "Tottenham Hotspur",
            "Wolves": "Wolverhampton Wanderers",
            "Arsenal": "Arsenal",
            "Sheffield United": "Sheffield United",
            "Burnley": "Burnley",
            "Southampton": "Southampton",
            "Everton": "Everton",
            "Newcastle": "Newcastle United",
            "Crystal Palace": "Crystal Palace",
            "Brighton": "Brighton & Hove Albion",
            "West Ham": "West Ham United",
            "Aston Villa": "Aston Villa",
            "Bournemouth": "AFC Bournemouth",
            "Watford": "Watford",
            "Norwich": "Norwich City",
        },
        2020: {
            "Liverpool": "Liverpool",
            "Man City": "Manchester City",
            "Man United": "Manchester United",
            "Chelsea": "Chelsea",
            "Leicester": "Leicester City",
            "Tottenham": "Tottenham Hotspur",
            "Wolves": "Wolverhampton Wanderers",
            "Arsenal": "Arsenal",
            "Sheffield United": "Sheffield United",
            "Burnley": "Burnley",
            "Southampton": "Southampton",
            "Everton": "Everton",
            "Newcastle": "Newcastle United",
            "Crystal Palace": "Crystal Palace",
            "Brighton": "Brighton & Hove Albion",
            "West Ham": "West Ham United",
            "Aston Villa": "Aston Villa",
            "Fulham": "Fulham",
            "Leeds": "Leeds United",
            "West Brom": "West Bromwich Albion",
        },
        2021: {
            "Liverpool": "Liverpool",
            "Man City": "Manchester City",
            "Man United": "Manchester United",
            "Chelsea": "Chelsea",
            "Leicester": "Leicester City",
            "Tottenham": "Tottenham Hotspur",
            "Wolves": "Wolverhampton Wanderers",
            "Arsenal": "Arsenal",
            "Watford": "Watford",
            "Burnley": "Burnley",
            "Southampton": "Southampton",
            "Everton": "Everton",
            "Newcastle": "Newcastle United",
            "Crystal Palace": "Crystal Palace",
            "Brighton": "Brighton & Hove Albion",
            "West Ham": "West Ham United",
            "Aston Villa": "Aston Villa",
            "Brentford": "Brentford",
            "Leeds": "Leeds United",
            "Norwich": "Norwich City",
        },
        2022: {
            "Liverpool": "Liverpool",
            "Man City": "Manchester City",
            "Man United": "Manchester United",
            "Chelsea": "Chelsea",
            "Leicester": "Leicester City",
            "Tottenham": "Tottenham Hotspur",
            "Wolves": "Wolverhampton Wanderers",
            "Arsenal": "Arsenal",
            "Nott'm Forest": "Nottingham Forest",
            "Fulham": "Fulham",
            "Southampton": "Southampton",
            "Everton": "Everton",
            "Newcastle": "Newcastle United",
            "Crystal Palace": "Crystal Palace",
            "Brighton": "Brighton & Hove Albion",
            "West Ham": "West Ham United",
            "Aston Villa": "Aston Villa",
            "Brentford": "Brentford",
            "Leeds": "Leeds United",
            "Bournemouth": "AFC Bournemouth",
        },
        2023: {
            "Liverpool": "Liverpool",
            "Man City": "Manchester City",
            "Man United": "Manchester United",
            "Chelsea": "Chelsea",
            "Luton": "Luton Town",
            "Tottenham": "Tottenham Hotspur",
            "Wolves": "Wolverhampton Wanderers",
            "Arsenal": "Arsenal",
            "Nott'm Forest": "Nottingham Forest",
            "Fulham": "Fulham",
            "Burnley": "Burnley",
            "Everton": "Everton",
            "Newcastle": "Newcastle United",
            "Crystal Palace": "Crystal Palace",
            "Brighton": "Brighton & Hove Albion",
            "West Ham": "West Ham United",
            "Aston Villa": "Aston Villa",
            "Brentford": "Brentford",
            "Sheffield United": "Sheffield United",
            "Bournemouth": "AFC Bournemouth",
        },
    }

    df["Season"] = pd.to_numeric(df["Season"], errors="coerce") # The values in the Season column are converted to numeric values with invalid values being null
    df["FifaVersion"] = df["Season"].map(season_to_fifa_version) # The Season for each row in df is mapped to the respective fifa version

    df["HomeTeamFifa"] = [season_team_name_map.get(season, {}).get(team, team) for team, season in zip(df["HomeTeam"], df["Season"])] # Each home team is paired with the respective season and the respective season is then found from the outer dictionary and from the inner dictionary the fifa team name is then returned for each pair in the form of a list
    df["AwayTeamFifa"] = [season_team_name_map.get(season, {}).get(team, team) for team, season in zip(df["AwayTeam"], df["Season"])] # Each away team is paired with the respective season and the respective season is then found from the outer dictionary and from the inner dictionary the fifa team name is then returned for each pair in the form of a list

    fifa = pd.read_csv("data/processed/fifa_20-24_teams_data.csv") # Fifa ratings csv file is loaded into a Dataframe
    fifa["fifa_version"] = pd.to_numeric(fifa["fifa_version"], errors="coerce") # The values in the fifa_version column are converted to numeric values with invalid values being null
    fifa = fifa[fifa["fifa_version"].isin(season_to_fifa_version.values())] # Dataframe is filtered using the previosly defined mapping
    fifa["Season"] = fifa["fifa_version"].map({v: k for k, v in season_to_fifa_version.items()}) # Reverses the mapping in season_to_fifa_version dictionary and a new mapping stored in the newly added Season column is created from Fifa version to Season
    fifa_ratings = fifa[["Season", "team_name", "overall", "attack", "midfield", "defence"]].copy() # The required columns are selected

    home_fifa_ratings = fifa_ratings.rename(columns={ # Creates a dataframe for the home ratings and renames the columns appropriately
        "team_name": "HomeTeamFifa",
        "overall": "HomeFifaOverall",
        "attack": "HomeFifaAttack",
        "midfield": "HomeFifaMidfield",
        "defence": "HomeFifaDefence",
    })
    away_fifa_ratings = fifa_ratings.rename(columns={ # Creates a dataframe for the away ratings and renames the columns appropriately
        "team_name": "AwayTeamFifa",
        "overall": "AwayFifaOverall",
        "attack": "AwayFifaAttack",
        "midfield": "AwayFifaMidfield",
        "defence": "AwayFifaDefence",
    })

    df = df.merge(home_fifa_ratings, how="left", on=["Season", "HomeTeamFifa"]) # Performs a left join from df to home_fifa_ratings matching Season and HomeTeamFifa
    df = df.merge(away_fifa_ratings, how="left", on=["Season", "AwayTeamFifa"]) # Performs a left join from df to away_fifa_ratings matching Season and AwayTeamFifa

    ratings_cols = [ # Columns to store in form csv file are selected
        "Season", "FifaVersion", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "HomeFifaOverall", "HomeFifaAttack", "HomeFifaMidfield", "HomeFifaDefence",
        "AwayFifaOverall", "AwayFifaAttack", "AwayFifaMidfield", "AwayFifaDefence",
    ]
    df_ratings = df[ratings_cols].dropna() # Null values are removed
    save_csv(df_ratings, "data/features/eng1_data_ratings.csv") # Csv without null values is saved into the specified directory

    # Combined dataset
    modelling_cols = [
        "Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded",
        "Bet365HomeWinOddsPercentage", "Bet365DrawOddsPercentage", "Bet365AwayWinOddsPercentage",
        "OddsDifference_HvA", "OddsDifference_HvD", "OddsDifference_AvD",
        "HomeForm", "AwayForm", "HomeAdvantageIndex",
        "HomeGeneralForm", "AwayGeneralForm", "GeneralFormDifference",
        "AverageGoalsScoredAtHome", "AverageGoalsScoredAtAway",
        "AverageGoalsConcededAtHome", "AverageGoalsConcededAtAway",
        "TotalGoalsScoredHome", "TotalGoalsScoredAway",
        "TotalGoalsConcededHome", "TotalGoalsConcededAway",
        "HistoricalEncountersHome", "HistoricalEncountersAway",
        "HomeFifaOverall", "HomeFifaAttack", "HomeFifaMidfield", "HomeFifaDefence",
        "AwayFifaOverall", "AwayFifaAttack", "AwayFifaMidfield", "AwayFifaDefence",
    ]
    df_combined = df[modelling_cols].fillna(0) # Sets missing values to 0 since form cannot be calculated on first appearance
    save_csv(df_combined, "data/features/eng1_data_combined.csv") # Csv is saved into specified directory

if __name__ == "__main__":
    prepare_features()
