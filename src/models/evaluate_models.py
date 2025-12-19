import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model_name, y_true, y_pred):
    labels = [1, 0, -1] # The labels are defined
    label_names = ["Home", "Draw", "Away"] # The labels are defined in the form of a list in a readable format
    accuracy = accuracy_score(y_true, y_pred) # The accuracy of the model is calculated by comparing the testing data with the predicted results
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0) # The precision, recall f1 score and support are computed for each class
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0) # The macro averaged precision, recall and f1 score are computed
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0) # The weighted averaged precision, recall and f1 score are computed

    # Dictionary creation to store the performance metrics
    metrics = {
        "Model": model_name, # Stores the name of the model being evaluated
        "Accuracy": accuracy, # Stores the accuracy of the model being evaluated
        "Precision_Home": precision[0], # Stores the precision for the home label
        "Recall_Home": recall[0], # Stores the recall for the home label
        "F1_Home": f1[0], # Stores the f1 score for the home label
        "Support_Home": support[0], # Stores the support for the home label
        "Precision_Draw": precision[1], # Stores the precision for the draw label
        "Recall_Draw": recall[1], # Stores the recall for the draw label
        "F1_Draw": f1[1], # Stores the f1 score for the draw label
        "Support_Draw": support[1], # Stores the support for the draw label
        "Precision_Away": precision[2], # Stores the precision for the away label
        "Recall_Away": recall[2], # Stores the recall for the away label
        "F1_Away": f1[2], # Stores the f1 score for the away label
        "Support_Away": support[2], # Stores the support for the away label
        "Precision_Macro": macro_precision, # Stores the macro precision of the model being evaluated
        "Recall_Macro": macro_recall, # Stores the macro recall of the model being evaluated
        "F1_Macro": macro_f1, # Stores the macro F1 Score of the model being evaluated
        "Precision_Weighted": weighted_precision, # Stores the weighted precision of the model being evaluated
        "Recall_Weighted": weighted_recall, # Stores the weighted recall of the model being evaluated
        "F1_Weighted": weighted_f1 # Stores the weighted F1 Score of the model being evaluated
    }

    print(f"\n{model_name} Performance:") # Prints the name of the model showing its performance
    print(f"  Accuracy: {accuracy:.3f}") # The accuracy is printed
    for name, p, r, f, s in zip(label_names, precision, recall, f1, support): # For each label tuple
        print(f"  {name}: Precision {p:.3f}  Recall {r:.3f}  F1 {f:.3f}  Support {s}") # Prints the precision, recall, f1 and support for the respective label
    print(f"  Macro Average: Precision {macro_precision:.3f}  Recall {macro_recall:.3f}  F1 {macro_f1:.3f}") # The macro precision, recall and f1 score are printed
    print(f"  Weighted Average: Precision {weighted_precision:.3f}  Recall {weighted_recall:.3f}  F1 {weighted_f1:.3f}") # The weighted precision, recall and f1 are printed
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
