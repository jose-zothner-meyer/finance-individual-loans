import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

class FeatureReduction:
    def __init__(self, df):
        """
        Initialize the FeatureReduction class with a DataFrame.

        Parameters:
        df (DataFrame): The input data frame containing features and target.
        """
        self.df = df

    def feature_reduction_pca(self, n_components=0.95):
        """
        Perform feature reduction using PCA. Retains enough components to explain n_components variance.
        
        Parameters:
        n_components (float): Desired variance percentage to retain.

        Returns:
        DataFrame: DataFrame with reduced features.
        """
        pca = PCA(n_components=n_components)
        numeric_features = self.df.select_dtypes(include=[np.number])
        reduced_data = pca.fit_transform(numeric_features)
        reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(reduced_data.shape[1])])
        print(f"Reduced dataset to {reduced_df.shape[1]} components, explaining {n_components * 100}% of variance.")
        return reduced_df

    def calculate_feature_importance(self, target_column):
        """
        Calculate feature importance using a RandomForestClassifier.

        Parameters:
        target_column (str): The name of the target column for classification.

        Returns:
        DataFrame: DataFrame of feature importances.
        """
        if target_column not in self.df.columns:
            raise ValueError(f"Target column {target_column} not found in the dataset.")
        
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        # Use only numeric features for RandomForestClassifier
        X_numeric = X.select_dtypes(include=[np.number])

        model = RandomForestClassifier(random_state=42)
        model.fit(X_numeric, y)
        importances = model.feature_importances_

        feature_importances = pd.DataFrame({
            'Feature': X_numeric.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        print("Feature Importance:\n", feature_importances)
        return feature_importances
