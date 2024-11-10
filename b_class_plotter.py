import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

class Plotter:
    def __init__(self, df):
        self.df = df

    def plot_null_distribution(self, initial_nulls, final_nulls):
        """Visualize null counts before and after handling nulls."""
        # Ensure 'Null Count' column exists
        if 'Null Count' not in initial_nulls.columns or 'Null Count' not in final_nulls.columns:
            raise ValueError("Null summaries should contain 'Null Count' as a column.")

        fig = go.Figure()

        # Initial null counts
        fig.add_trace(go.Bar(
            x=initial_nulls['index'],
            y=initial_nulls['Null Count'],
            name='Initial Null Count',
            marker=dict(color='rgba(255, 99, 71, 0.6)')
        ))

        # Final null counts
        fig.add_trace(go.Bar(
            x=final_nulls['index'],
            y=final_nulls['Null Count'],
            name='Final Null Count',
            marker=dict(color='rgba(100, 149, 237, 0.6)')
        ))

        fig.update_layout(
            title="Null Value Distribution Before and After Transformation",
            xaxis_title="Columns",
            yaxis_title="Null Count",
            barmode='group'
        )
        fig.show()

    def plot_numerical_distributions(self):
        """Plot histograms for all numerical columns."""
        num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            fig = px.histogram(self.df, x=col, nbins=30, title=f"Distribution of {col}")
            fig.show()

    def plot_categorical_counts(self):
        """Plot bar charts for all categorical columns with green color."""
        cat_cols = self.df.select_dtypes(include=['category']).columns
        for col in cat_cols:
            # Count unique values and prepare data
            value_counts = self.df[col].value_counts().reset_index()
            value_counts.columns = [col, 'count']
            
            # Create bar chart with green color
            fig = px.bar(
                value_counts, 
                x=col, 
                y='count', 
                title=f"Counts of {col}",
                color_discrete_sequence=['#32CD32']  # Lime green
            )
            
            # Update axis labels and show figure
            fig.update_layout(xaxis_title=col, yaxis_title="Count")
            fig.show()
            
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

    def plot_histograms_with_kde(self, columns, col_wrap=3, height=3):
        """Generates histograms with KDE for each column using Seaborn's FacetGrid."""
        # Creating a subset DataFrame with the specified columns
        subset_df = self.df[columns].replace([np.inf, -np.inf], np.nan).dropna()

        # Melt the DataFrame to long-form format for FacetGrid
        melted_df = subset_df.melt(var_name="Column", value_name="Value")

        # Create a FacetGrid for the melted DataFrame
        g = sns.FacetGrid(melted_df, col="Column", col_wrap=col_wrap, height=height, sharex=False, sharey=False)
        g.map(sns.histplot, "Value", kde=True)

        # Set the titles and layout properly
        g.set_titles(col_template="{col_name}")
        plt.subplots_adjust(top=0.9)  # Adjust the top of the plot to make space for the suptitle
        g.fig.suptitle("Histograms with KDE for Specified Columns")
        plt.show()
        
    # Corrected method for box and whisker plot visualization
    def plot_box_plots(self, columns, title_prefix="Box and Whisker Plot"):
        """
        Plots box and whisker plots for the given columns in the dataframe.
        
        Args:
            columns (list of str): The columns to visualize using box plots.
            title_prefix (str): The prefix to use in the title of each plot.
        """
        for column in columns:
            if column in self.df.columns:
                fig = px.box(self.df, y=column, title=f"{title_prefix} - {column}")
                fig.show()

    def visualise_correlation_matrix(self):
        """
        Visualize the correlation matrix of the numeric features in the DataFrame.
        """
        # Select only numeric columns to avoid errors during correlation computation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_df = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce')  # Coerce any non-numeric values to NaN

        # Compute the correlation matrix
        correlation_matrix = numeric_df.corr()

        # Plot the heatmap using Seaborn
        plt.figure(figsize=(20, 15))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
        plt.title("Correlation Matrix of the Numeric Features in the Dataset")
        plt.show()

    def plot_feature_importance(self, feature_importances):
        """
        Visualize feature importance using a bar chart.

        Parameters:
        feature_importances (DataFrame): DataFrame containing feature names and their importance scores.
        """
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importances, hue='Feature', palette='viridis', dodge=False, legend=False)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.show()