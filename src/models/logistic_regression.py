import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.models.evaluate_models import evaluate_model, save_results

def run_logistic_regression(data_path="data/features/eng1_data_combined.csv"):
    df = pd.read_csv(data_path) # Loads dataset

    train_df = df[df["Season"] < 2023] # Splits training data using 2019 - 2022 Seasons
    test_df = df[df["Season"] == 2023] # Splits testing data using 2023 Season only

    # Features used to predict are defined
    features = [
        "Bet365HomeWinOdds",
        "Bet365DrawOdds",
        "Bet365AwayWinOdds",
        "OddsDifference_Bet365"
    ]

    X_train = train_df[features] # Features used for training
    y_train = train_df["ResultEncoded"] # Results used for training
    X_test = test_df[features] # Features used for testing
    y_test = test_df["ResultEncoded"] # Results used for testing

    model = LogisticRegression(max_iter=500) # Logistic Regression model is instantiated with the maximum number of iterations being set to 500
    model.fit(X_train, y_train) # Model is trained

    preds = model.predict(X_test) # Predicted outcomes are stored

    results = evaluate_model("Logistic Regression", y_test, preds) # Model performance is evaluated and stored in results
    save_results(results) # Model results are saved

    out = test_df.copy() # It stored a copy of the dataframe used for testing in out
    out["Predicted"] = preds # Add a predicted column to the dataframe
    out.to_csv("data/results/logistic_regression_2023_predictions.csv", index=False) # Converts the final dataframe to csv and appends it to existing csv or stores it in new csv
    print("Predictions saved to: data/results/logistic_regression_2023_predictions.csv") # Prints confirmation that the results have been stored
    return model, results # Returns the trained model and the respective results

if __name__ == "__main__":
    run_logistic_regression()