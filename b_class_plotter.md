
---

# Code Breakdown and Explanation: `Plotter` Class

The `Plotter` class provides various visualization tools for exploring and understanding the dataset. It utilizes a combination of Plotly, Matplotlib, and Seaborn to generate different types of visualizations. Below is a detailed breakdown of the class and its methods.

### Code

```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
```
- **Import necessary libraries**:
  - `plotly.express` and `plotly.graph_objects` for interactive visualizations.
  - `pandas` for data manipulation.
  - `matplotlib.pyplot` and `seaborn` for static plotting.
  - `scipy.stats` for generating Q-Q plots.
  - `numpy` for numerical operations.

```python
class Plotter:
```
- **Defines the `Plotter` class** which contains multiple methods to generate visualizations based on the provided DataFrame.

```python
    def __init__(self, df):
        self.df = df
```
- **Initializes the `Plotter` class** with a pandas DataFrame (`df`) which will be used for all subsequent plotting methods.

```python
    def plot_null_distribution(self, initial_nulls, final_nulls):
        """Visualize null counts before and after handling nulls."""
        # Ensure 'Null Count' column exists
        if 'Null Count' not in initial_nulls.columns or 'Null Count' not in final_nulls.columns:
            raise ValueError("Null summaries should contain 'Null Count' as a column.")
```
- **Defines the `plot_null_distribution` method** to visualize the number of missing values before and after handling nulls.
  - **Checks that the input DataFrames** (`initial_nulls` and `final_nulls`) contain a column named `'Null Count'` to avoid errors.

```python
        fig = go.Figure()

        # Initial null counts
        fig.add_trace(go.Bar(
            x=initial_nulls['index'],
            y=initial_nulls['Null Count'],
            name='Initial Null Count',
            marker=dict(color='rgba(255, 99, 71, 0.6)')
        ))
```
- **Adds a bar trace to the figure** (`fig`) to represent the initial null counts, using a specific color to distinguish it.

```python
        # Final null counts
        fig.add_trace(go.Bar(
            x=final_nulls['index'],
            y=final_nulls['Null Count'],
            name='Final Null Count',
            marker=dict(color='rgba(100, 149, 237, 0.6)')
        ))
```
- **Adds a second bar trace** to represent the final null counts after processing, using a different color to compare before and after values.

```python
        fig.update_layout(
            title="Null Value Distribution Before and After Transformation",
            xaxis_title="Columns",
            yaxis_title="Null Count",
            barmode='group'
        )
        fig.show()
```
- **Updates the layout of the plot** to include titles for better understanding, and **sets the bar mode to `'group'`** for easy comparison. **Displays the plot**.

```python
    def plot_numerical_distributions(self):
        """Plot histograms for all numerical columns."""
        num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            fig = px.histogram(self.df, x=col, nbins=30, title=f"Distribution of {col}")
            fig.show()
```
- **Defines the `plot_numerical_distributions` method** to create histograms for all numeric columns.
  - **Iterates through each numeric column** and generates a histogram using Plotly (`px.histogram`) to visualize the distribution of data.

```python
    def plot_categorical_counts(self):
        """Plot bar charts for all categorical columns with green color."""
        cat_cols = self.df.select_dtypes(include=['category']).columns
        for col in cat_cols:
            value_counts = self.df[col].value_counts().reset_index()
            value_counts.columns = [col, 'count']
            
            fig = px.bar(
                value_counts, 
                x=col, 
                y='count', 
                title=f"Counts of {col}",
                color_discrete_sequence=['#32CD32']  # Lime green
            )
            
            fig.update_layout(xaxis_title=col, yaxis_title="Count")
            fig.show()
```
- **Defines the `plot_categorical_counts` method** to create bar charts for all categorical columns.
  - Generates **bar charts** with counts of each category in the column, using a **green color scheme** for visual consistency.

```python
    def qq_plot(self, column, figsize=(8, 6)):
        """Generates a Q-Q plot for the specified column."""
        plt.figure(figsize=figsize)
        data = self.df[column].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 10:
            print(f"Not enough data to generate Q-Q plot for {column}")
            return
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot of {column}")
        plt.show()
```
- **Defines the `qq_plot` method** to generate Q-Q plots to visualize the normality of the data in a specific column.
  - Handles **infinite values and NaNs** to avoid errors. If there are fewer than ten data points, **prints a warning** and exits.

```python
    def plot_histograms_with_kde(self, columns, col_wrap=3, height=3):
        """Generates histograms with KDE for each column using Seaborn's FacetGrid."""
        subset_df = self.df[columns].replace([np.inf, -np.inf], np.nan).dropna()
        melted_df = subset_df.melt(var_name="Column", value_name="Value")

        g = sns.FacetGrid(melted_df, col="Column", col_wrap=col_wrap, height=height, sharex=False, sharey=False)
        g.map(sns.histplot, "Value", kde=True)

        g.set_titles(col_template="{col_name}")
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle("Histograms with KDE for Specified Columns")
        plt.show()
```
- **Defines the `plot_histograms_with_kde` method** to generate histograms with KDE (Kernel Density Estimate) using Seaborn's `FacetGrid`.
  - **Melts the DataFrame** to create a long-form structure suitable for `FacetGrid` and plots histograms with KDE overlays for each column.

```python
    def plot_box_plots(self, columns, title_prefix="Box and Whisker Plot"):
        """Plots box and whisker plots for the given columns in the dataframe."""
        for column in columns:
            if column in self.df.columns:
                fig = px.box(self.df, y=column, title=f"{title_prefix} - {column}")
                fig.show()
```
- **Defines the `plot_box_plots` method** to generate box and whisker plots for specified columns.
  - **Uses Plotly** to create interactive box plots for each column, with a customizable title prefix.

```python
    def visualise_correlation_matrix(self):
        """Visualize the correlation matrix of the numeric features in the DataFrame."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_df = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        correlation_matrix = numeric_df.corr()

        plt.figure(figsize=(20, 15))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
        plt.title("Correlation Matrix of the Numeric Features in the Dataset")
        plt.show()
```
- **Defines the `visualise_correlation_matrix` method** to visualize correlations between numeric features.
  - **Computes the correlation matrix** and plots it as a **heatmap** using Seaborn to visually indicate the strength of relationships between features.

```python
    def plot_feature_importance(self, feature_importances):
        """Visualize feature importance using a bar chart."""
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importances, hue='Feature', palette='viridis', dodge=False, legend=False)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.show()
```
- **Defines the `plot_feature_importance` method** to visualize feature importance as a bar chart.
  - **Uses Seaborn** to create a bar plot of feature importances, indicating which features are most influential in a given model.

---

### Summary
- The `Plotter` class provides a comprehensive suite of methods for **visualizing data distributions, categorical counts, null values, and feature importance**.
- **Interactive plots (using Plotly)**:
  - `plot_null_distribution`: Compare null values before and after processing.
  - `plot_numerical_distributions`: Show histograms of numerical data.
  - `plot_categorical_counts` and `plot_box_plots`: Plot bar charts and box plots for categorical data.
- **Static plots (using Matplotlib and Seaborn)**:
  - `qq_plot`: Generate Q-Q plots for normality assessment.
  - `plot_histograms_with_kde`: Show histograms with KDE overlays.
  - `visualise_correlation_matrix`: Visualize correlations between features.
  - `plot_feature_importance`: Show feature importance using a bar plot.
  
The combination of Plotly, Matplotlib, and Seaborn allows for both **static and interactive visualizations**, catering to different analysis needs.