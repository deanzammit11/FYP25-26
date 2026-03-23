from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import ensure_dirs

def pca_feature_analysis():
    ensure_dirs("data/results/pca") # Checks if the PCA results directory exists and if it does not it creates it
    output_path = Path("data/results/pca") # The PCA output directory path is stored

    df = pd.read_csv("data/features/eng1_data_combined.csv") # The features dataset is stored in a dataframe

    excluded_columns = {"Season", "Date", "HomeTeam", "AwayTeam", "ResultEncoded"} # Columns which are not features are stored in a set
    feature_columns = [column for column in df.columns if column not in excluded_columns and is_numeric_dtype(df[column])] # Each column which is not in the list of excluded columns and is also numeric is stored in a feature columns list
    if not feature_columns: # If the feature columns list is empty
        raise ValueError("No numeric feature columns were found for PCA analysis.") # A value error is raised

    feature_df = df[feature_columns].copy() # The feature columns are copied into a new dataframe
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce") # Each feature column is converted to numeric with any invalid values converted to null values
    feature_df = feature_df.fillna(0.0) # Any missing values are replaced with 0

    n_samples, n_features = feature_df.shape # The number of rows representing the number of matches and the number of columns representing the number of features in the features dataframe are stored seperately
    if n_samples < 2: # If the dataset contains less than 2 rows
        raise ValueError("At least two rows are required for PCA analysis.") # A value error is raised

    n_components = min(n_samples, n_features) # The number of maximum components is set as the smaller value between the number of rows and the number of features
    if n_components < 1: # If the number of components is less than 1
        raise ValueError("At least 1 component is required for PCA analysis.") # A value error is raised

    scaled_features = StandardScaler().fit_transform(feature_df) # The feature dataframe is scaled so each feature has mean 0 and a standard deviation 1

    pca = PCA(n_components=n_components) # A PCA object which computes the maximum number of components is created
    scores = pca.fit_transform(scaled_features) # PCA is fitted to the scaled features and the transformed component scores for each row are built

    component_labels = [f"PC{i}" for i in range(1, n_components + 1)] # A list containing the label of each principal component is built

    explained_variance_df = pd.DataFrame( # A dataframe storing the explained variance information for each component is created
        { # The explained variance columns are defined in a dictionary
            "Component": component_labels, # The principal component labels are stored
            "ExplainedVarianceRatio": pca.explained_variance_ratio_, # The ratio of total variance explained by each component is stored
            "CumulativeExplainedVarianceRatio": pca.explained_variance_ratio_.cumsum(), # The cumulative explained variance across components is stored
            "Eigenvalue": pca.explained_variance_ # The eigenvalue of each component is stored
        }
    )
    explained_variance_df.to_csv(output_path / "pca_explained_variance.csv", index=False) # The csv is saved to the specified directory

    loadings_df = pd.DataFrame( # A dataframe storing the component loadings showing the contribution of each feature to the principal component along with the direction is created
        pca.components_.T, # The principal component weights matrix is transposed so each row represents one original feature
        index=feature_columns, # The name of the features are used as the row index
        columns=component_labels, # The lables of the principal components are used as the dataframe columns
    ).reset_index(names="Feature") # The feature index is converted into a normal column named Feature
    loadings_df.to_csv(output_path / "pca_component_loadings.csv", index=False) # The csv is saved to the specified directory

    absolute_loadings_df = ( # A dataframe storing the absolute size of each component loading showing the contribution of each feature to the principal component is created
        loadings_df.set_index("Feature") # The initial loadings datafrane is used and the Feature column is set as the index
        .abs() # The absolute value of each numeric loading value is taken
        .reset_index(names="Feature") # The feature index is converted back into a normal column named Feature
    )
    absolute_loadings_df.to_csv(output_path / "pca_component_loadings_absolute.csv", index=False) # The csv is saved to the specified directory

    dominant_component_rows = [] # An empty list where each row will store the name of the feature and with which principal component it is the most associated with along with the value is initialised
    absolute_loadings_indexed = absolute_loadings_df.set_index("Feature") # The absolute loadings dataframe is indexed by Feature
    for feature_name in absolute_loadings_indexed.index: # For each feature name in the indexed absolute loadings dataframe
        feature_loadings = absolute_loadings_indexed.loc[feature_name] # The row corresponding to the current feature containing the absolute loading of that feature for each principal component is selected
        dominant_component = feature_loadings.idxmax() # The column name with the largest value in the resulting in the respective component label is selected
        dominant_component_rows.append( # A dictionary containing the information related to the dominant component for the respective feature is appended to the list
            {
                "Feature": feature_name, # The current feature name is stored
                "DominantComponent": dominant_component, # The component with the highest absolute loading is stored
                "AbsoluteLoading": float(feature_loadings[dominant_component]), # The size of that highest absolute loading is stored as a float
            }
        )
    dominant_components_df = pd.DataFrame(dominant_component_rows) # The dominant component rows are converted into a dataframe
    dominant_components_df["DominantComponentNumber"] = dominant_components_df["DominantComponent"].str.replace("PC", "", regex=False).astype(int) # The numeric part of the dominant component label is extracted and stored as an integer for sorting
    dominant_components_df = dominant_components_df.sort_values(["DominantComponentNumber", "AbsoluteLoading", "Feature"], ascending=[True, False, True]).drop(columns=["DominantComponentNumber"]) # The dominant components dataframe is sorted by component number in ascending order, then by loading size in descending order, then by feature name in alphabetical order and finally the component number column used for sorting is dropped
    dominant_components_df.to_csv(output_path / "pca_dominant_component_by_feature.csv", index=False) # The csv is saved to the specified directory

    score_columns = component_labels # All component labels are stored in a new list
    scores_df = pd.DataFrame(scores, columns=score_columns) # The scores array is converted into a dataframe with the component labels being the columns
    scores_df.insert(0, "RowNumber", range(1, len(scores_df) + 1)) # A RowNumber column starting from 1 up to the final match is inserted as the first column of the scores dataframe
    scores_df.to_csv(output_path / "pca_scores_all_components.csv", index=False) # The csv is saved to the specified directory

    top_loading_rows = [] # An empty list which will store one dataframe for each principal component each containing the top 10 strongest feeatures for that component is initialised
    for component in component_labels: # For each principal component label
        ordered = loadings_df[["Feature", component]].copy() # The Feature column and the current component column containing the feature names and the loading values for each feature are copied into a new dataframe
        ordered["AbsoluteLoading"] = ordered[component].abs() # The absolute loading for the current component is computed and is stored in a new column
        ordered = ordered.sort_values(["AbsoluteLoading", "Feature"], ascending=[False, True]).head(10) # The current component loadings are sorted by the absolute loading value in descending order and then by Feature in alphabetical order and only the top 10 rows are kept
        ordered.insert(0, "Component", component) # A new column containing the current component lable is inserted as the first column of the ordered dataframe
        ordered = ordered[["Component", "Feature", "AbsoluteLoading", component]] # The columns are ordered
        top_loading_rows.append(ordered) # The current top loadings dataframe is appended to the list
    pd.concat(top_loading_rows, ignore_index=True).to_csv(output_path / "pca_top_10_features_per_component.csv", index=False) # All the top loadings dataframes are concatenated with the row index being reset and the csv is saved to the specified directory

    if "ResultEncoded" in df.columns: # If the ResultEncoded column is present in the dataframe
        result_labels = (pd.to_numeric(df["ResultEncoded"], errors="coerce").map({1: "Home Win", 0: "Draw", -1: "Away Win"}).fillna("Unknown")) # The ResultEncoded column is converted to numeric with invalid values becoming null and each value is mapped to a label with nulls being replaced with Unknown
        plot_component_pairs = [("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")] # The principal component pairs for which scatter plots will be plotted are defined
        colour_map = { # The colours used for each result label in the scatter plots are defined
            "Home Win": "#d62728", # Home Wins are coloured red
            "Draw": "#2ca02c", # Draws are coloured green
            "Away Win": "#1f77b4", # Away Wins are coloured blue
            "Unknown": "#7f7f7f" # Unknowns are coloured grey
        }

        for x_component, y_component in plot_component_pairs: # For each x and y component pair
            if x_component not in component_labels or y_component not in component_labels: # If the label of either component x or component y does not exist
                continue # The current component pair is skipped and it moves on to the next component pair

            pair_plot_df = pd.DataFrame( # A dataframe for the current pair scatter plot is built
                { # The current pair plot columns are defined in a dictionary
                    x_component: scores[:, component_labels.index(x_component)], # The position of x_component in the component_labels list is found and all the rows for that principal component column containing the PCA score for each match are stored in a seperate column in the dataframe
                    y_component: scores[:, component_labels.index(y_component)], # The position of y_component in the component_labels list is found and all the rows for that principal component column containing the PCA score for each match are stored in a seperate column in the dataframe
                    "ResultLabel": result_labels # The readable result match label is added to each row in the dataframe in a new ResultLabel column
                }
            )

            fig, ax = plt.subplots(figsize=(10, 7)) # A matplotlib figure and axis are created for the current PCA scatter plot
            for result_label in ["Home Win", "Draw", "Away Win", "Unknown"]: # For each result label in the defined order
                label_df = pair_plot_df[pair_plot_df["ResultLabel"] == result_label] # The pair plot dataframe is filtered for rows of the current result label
                if label_df.empty: # If no rows exist for the current result label
                    continue # The current result label is skipped and it moves on to the next label
                ax.scatter( # A scatter plot layer for the current result label is drawn
                    label_df[x_component], # The x-axis component values for the current label are plotted
                    label_df[y_component], # The y-axis component values for the current label are plotted
                    s=28, # The point size is set to 28
                    alpha=0.75, # The point transparency is set to 0.75
                    label=result_label, # The legend label is set to the current result label
                    color=colour_map[result_label] # The point colour is selected from the colour map
                )

            ax.set_title(f"PCA Scatter Plot: {x_component} vs {y_component}") # The title for the scatter plot is set
            ax.set_xlabel(x_component) # The x-axis label for the scatter plot is set
            ax.set_ylabel(y_component) # The y-axis label for the scatter plot is set
            ax.grid(True, alpha=0.3) # Grid lines of transparency 0.3 are added across the y-axis
            ax.legend() # The legend showing the result labels is displayed
            fig.tight_layout() # Layout is adjusted
            fig.savefig(output_path / f"pca_scatter_{x_component.lower()}_{y_component.lower()}_by_result.png", dpi=200, bbox_inches="tight") # The current PCA scatter plot is saved to the specified path setting the resolution and cropping around the table removing extra whitespace
            plt.close(fig) # Current figure is closed

    component_numbers = list(range(1, len(explained_variance_df) + 1)) # A list of principal component numbers starting from 1 is created for plotting
    x_tick_step = max(1, len(component_numbers) // 10) # The spacing between x-axis ticks is set by selecting the largest step between 1 and the whole number returned when dividing the number of components by 10
    x_tick_positions = component_numbers[::x_tick_step] # The x-axis tick positions are stored by stepping through the list by the computed step size
    if x_tick_positions[-1] != component_numbers[-1]: # If the last component number is not already included in the tick positions
        x_tick_positions.append(component_numbers[-1]) # The last component number is appended to the list of tick positions so the end of the axis is labelled

    fig, ax = plt.subplots(figsize=(10, 6)) # A matplotlib figure and axis are created for the scree plot
    ax.plot(component_numbers, explained_variance_df["ExplainedVarianceRatio"], marker="o", linewidth=2) # A line for the scree plot with the component numbers on the x-axis, explained variance ratio on the y-axis, a marker on each point and a line thickness of 2
    ax.set_title("PCA Scree Plot") # The title for the scree plot is set
    ax.set_xlabel("Principal Component") # The x-axis label for the scree plot is set
    ax.set_ylabel("Explained Variance Ratio") # The y-axis label for the scree plot is set
    ax.set_xticks(x_tick_positions) # The principal components are marked on the x-axis
    ax.grid(True, alpha=0.3) # Grid lines of transparency 0.3 are added across the y-axis
    fig.tight_layout() # Layout is adjusted
    fig.savefig(output_path / "pca_scree_plot.png", dpi=200, bbox_inches="tight") # The scree plot is saved to the specified path setting the resolution and cropping around the table removing extra whitespace
    plt.close(fig) # Figure is closed

    cumulative_fig, cumulative_ax = plt.subplots(figsize=(10, 6)) # A matplotlib figure and axis are created for the cumulative explained variance plot
    cumulative_ax.plot(component_numbers, explained_variance_df["CumulativeExplainedVarianceRatio"], marker="o", linewidth=2, color="#2ca02c") # A green line for the cumulative explained variance plot with the component numbers on the x-axis, cumulative explained variance ratio on the y-axis, a marker on each point and a line thickness of 2
    cumulative_ax.set_title("PCA Cumulative Explained Variance") # The title for the cumulative variance plot is set
    cumulative_ax.set_xlabel("Principal Component") # The x-axis label for the cumulative variance plot is set
    cumulative_ax.set_ylabel("Cumulative Explained Variance Ratio") # The y-axis label for the cumulative variance plot is set
    cumulative_ax.set_xticks(x_tick_positions) # The principal components are marked on the x-axis
    cumulative_ax.set_ylim(0, 1.05) # The limits of the y-axis are set from 0 to 1
    cumulative_ax.grid(True, alpha=0.3) # Grid lines of transparency 0.3 are added across the y-axis
    cumulative_fig.tight_layout() # Layout is adjusted
    cumulative_fig.savefig(output_path / "pca_cumulative_explained_variance.png", dpi=200, bbox_inches="tight") # The cumulative variance plot is saved to the specified path setting the resolution and cropping around the table removing extra whitespace
    plt.close(cumulative_fig) # Figure is closed

    heatmap_components = min(6, len(component_labels)) # The number of components to display in the loadings heatmap is the smallest value between 6 components and all the components
    heatmap_df = loadings_df.set_index("Feature")[component_labels[:heatmap_components]] # A dataframe which is indexed by feature containing the loadings for each respective feature for each heatmap component is built
    heatmap_fig_height = max(8, len(feature_columns) * 0.28) # The heatmap figure height is largest value between 8 and the number of features * 0.28
    heatmap_fig, heatmap_ax = plt.subplots(figsize=(10, heatmap_fig_height)) # A matplotlib figure and axis are created for the cumulative explained variance plot
    im = heatmap_ax.imshow(heatmap_df.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1) # The heatmap image is drawn using the loadings values with a fixed colour scale from -1 to 1
    heatmap_ax.set_title("PCA Component Loadings Heatmap") # The title for the heatmap is set
    heatmap_ax.set_xlabel("Principal Component") # The heatmap x-axis label is set
    heatmap_ax.set_ylabel("Feature") # The heatmap y-axis label is set
    heatmap_ax.set_xticks(range(heatmap_components)) # The principal components are marked on the x-axis
    heatmap_ax.set_xticklabels(component_labels[:heatmap_components]) # The labels for the x-axis ticks are set to the principal component names
    heatmap_ax.set_yticks(range(len(heatmap_df.index))) # The features are marked on the y-axis
    heatmap_ax.set_yticklabels(heatmap_df.index) # The labels for the y-axis ticks are set to the feature names
    heatmap_fig.colorbar(im, ax=heatmap_ax, label="Loading") # A bar describing loading magnitude based on the colour is added to the heatmap figure
    heatmap_fig.tight_layout() # Layout is adjusted
    heatmap_fig.savefig(output_path / "pca_component_loadings_heatmap.png", dpi=200, bbox_inches="tight") # The heatmap image is saved to the specified path setting the resolution and cropping around the table removing extra whitespace
    plt.close(heatmap_fig) # Figure is closed

    print(f"PCA explained variance saved to: {output_path / 'pca_explained_variance.csv'}") # A confirmation message showing where the explained variance csv was saved is printed
    print(f"PCA component loadings saved to: {output_path / 'pca_component_loadings.csv'}") # A confirmation message showing where the component loadings csv was saved is printed
    print(f"PCA plots saved under: {output_path}") # A confirmation message showing where the PCA plots were saved is printed

if __name__ == "__main__":
    pca_feature_analysis()