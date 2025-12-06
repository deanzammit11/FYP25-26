import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model_name, y_true, y_pred):
    # Dictionary creation to store the performance metrics
    metrics = {
        "Model": model_name, # Stores the name of the model being evaluated
        "Accuracy": accuracy_score(y_true, y_pred), # Stores the accuracy of the model being evaluated
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0), # Stores the precision of the model being evaluated
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0), # Stores the recall of the model being evaluated
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0) # Stores the F1 Score of the model being evaluated
    }

    print(f"\n{model_name} Performance:") # Prints the name of the model showing its performance
    for k, v in metrics.items(): # Loops through the metrics dictionary to display each metric
        if k != "Model": # Skips printing the name of the model only showing the metrics themselves since it was already printed seperately
            print(f"  {k}: {v:.3f}") # Metric is printed correct to 3 decimal places
    return metrics # Dictionary of metrics is returned

def save_results(results, output_path="data/results/model_metrics.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # Checks if directory for output file exists and if not it creates it
    df = pd.DataFrame([results]) # Converts the metrics dictionary into a single row pandas dataframe
    numeric_columns = df.select_dtypes(include="number").columns # The columns which have number as their dtype are picked
    df[numeric_columns] = df[numeric_columns].round(3) # The numeric columns are rounded correcto to 3 decimal places

    try: # Tries to read the existing csv as a pandas dataframe and if it exists it appends the new row to the existing data
        existing = pd.read_csv(output_path)
        df = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError: # If no existing file is found a new file is created
        pass

    df.to_csv(output_path, index=False) # Converts it to csv and appends it to existing csv or stores it in new csv
    print(f"Results saved to: {output_path}") # Message showing directory where results were saved
