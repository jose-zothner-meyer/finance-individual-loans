import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from scipy import stats  # For Box-Cox transformation
from scipy.stats import zscore, boxcox, yeojohnson
from b_class_data_transform import DataTransform
import random

class DataFrameTransform:
    def __init__(self, df):
        self.data_transform = DataTransform(df)
        self.df = df

    def check_nulls(self):
        """Display count and percentage of NULL values in each column."""
        null_counts = self.df.isnull().sum()
        null_percent = (null_counts / len(self.df)) * 100
        null_summary = pd.DataFrame({
            'Null Count': null_counts,
            'Percentage': null_percent
        })
        print("NULL Values Summary:\n", null_summary)
        return null_summary
    
    def drop_columns(self, threshold=50.0):
        """Drop columns with a percentage of NULL values above the threshold."""
        to_drop = self.df.columns[self.df.isnull().mean() * 100 > threshold]
        self.df.drop(columns=to_drop, inplace=True)
        print(f"\nDropped columns with more than {threshold}% missing data: {list(to_drop)}")
    
    def impute_missing_values(self, method='mean'):
        """Impute missing values in numerical columns with mean/median and categorical with mode."""
        for column in self.df.columns:
            if self.df[column].isnull().any():
                if self.df[column].dtype in ['float64', 'int64']:
                    if method == 'mean':
                        self.df[column].fillna(self.df[column].mean(), inplace=True)
                    elif method == 'median':
                        self.df[column].fillna(self.df[column].median(), inplace=True)
                elif self.df[column].dtype.name == 'category' or self.df[column].dtype == 'object':
                    if not self.df[column].mode().empty:  # Check for available mode
                        self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                    else:
                        self.df[column].fillna("Unknown", inplace=True)  # Default if no mode exists
        print(f"\nImputed missing values using the {method} method for numerical and mode for categorical columns.")
    
    def confirm_no_nulls(self):
        """Confirm that all NULL values have been removed."""
        null_summary = self.df.isnull().sum().sum()
        print(f"\nTotal remaining NULL values in DataFrame: {null_summary}")
        if null_summary == 0:
            print("All NULL values have been successfully removed or imputed.")
        else:
            print("There are still NULL values in the DataFrame.")
        return null_summary

    def transform_dataframe(self):
        # Apply label encoding and one-hot encoding using the DataTransform class
        self.data_transform.transform()

        # Return the transformed DataFrame
        return self.data_transform.df

    def get_label_mappings(self):
        """Get the label mappings after applying label encoding."""
        return self.data_transform.get_label_mappings()

    def identify_skewed_columns(self, threshold=0.5):
        # Identify columns with skewness above the threshold
        skewed_cols = self.df.select_dtypes(include=['float64', 'int64']).apply(lambda x: x.skew()).sort_values(ascending=False)
        skewed_cols = skewed_cols[abs(skewed_cols) > threshold]
        return skewed_cols.index.tolist()

    def log_transform(self, col):
        """Log transformation (handles non-positive values by shifting)."""
        try:
            if (self.df[col] <= 0).any():
                shifted_col = self.df[col] - self.df[col].min() + 1
                transformed = np.log1p(shifted_col)
            else:
                transformed = np.log1p(self.df[col])
            skew = transformed.skew()
            return {'transformed': transformed, 'skew': skew}
        except Exception as e:
            print(f"Log transformation failed on '{col}': {e}")
            return None

    def sqrt_transform(self, col):
        """Square root transformation (handles non-positive values by shifting)."""
        try:
            if (self.df[col] < 0).any():
                shifted_col = self.df[col] - self.df[col].min()
                transformed = np.sqrt(shifted_col)
            else:
                transformed = np.sqrt(self.df[col])
            skew = transformed.skew()
            return {'transformed': transformed, 'skew': skew}
        except Exception as e:
            print(f"Square Root transformation failed on '{col}': {e}")
            return None

    def reciprocal_transform(self, col):
        """Reciprocal transformation (handles zero values by adding a small constant)."""
        try:
            shifted_col = self.df[col] + 1e-6  # Adding a small constant to avoid division by zero
            transformed = 1 / shifted_col
            skew = transformed.skew()
            return {'transformed': transformed, 'skew': skew}
        except Exception as e:
            print(f"Reciprocal transformation failed on '{col}': {e}")
            return None

    def boxcox_transform(self, col):
        """Box-Cox transformation (only for positive values)."""
        try:
            if (self.df[col] > 0).all():
                transformed, _ = boxcox(self.df[col].dropna())
                transformed_series = pd.Series(transformed, index=self.df[col].dropna().index)
                skew = transformed_series.skew()
                return {'transformed': transformed_series, 'skew': skew}
            else:
                raise ValueError("Box-Cox transformation requires all positive values.")
        except Exception as e:
            print(f"Box-Cox transformation failed on '{col}': {e}")
            return None
        
    def yeojohnson_transform(self, col):
        """Yeo-Johnson transformation (works with any range of values)."""
        try:
            yeo_transformer = PowerTransformer(method='yeo-johnson')
            transformed = yeo_transformer.fit_transform(self.df[[col]].dropna())
            transformed_series = pd.Series(transformed.flatten(), index=self.df[col].dropna().index)
            skew = transformed_series.skew()
            return {'transformed': transformed_series, 'skew': skew}
        except Exception as e:
            print(f"Yeo-Johnson transformation failed on '{col}': {e}")
            return None

    def transform_skewed_columns(self, columns, skew_threshold=0.1, margin=0.05):
        """Apply transformations to skewed columns to reduce skewness, with flexibility in selection."""
        for col in columns:
            original_skew = self.df[col].skew()
            print(f"\nOriginal skewness for '{col}': {original_skew:.2f}")
            
            # Initialize a dictionary to store results of transformations
            transformations = {}

            # Apply different transformations
            transformations['boxcox'] = self.boxcox_transform(col)
            transformations['yeojohnson'] = self.yeojohnson_transform(col)
            transformations['log'] = self.log_transform(col)
            transformations['sqrt'] = self.sqrt_transform(col)
            transformations['reciprocal'] = self.reciprocal_transform(col)

            # Filter out any failed transformations
            transformations = {k: v for k, v in transformations.items() if v is not None}

            # Select the transformation with the lowest skew
            if transformations:
                # Get all transformations sorted by absolute skewness
                sorted_transformations = sorted(transformations.items(), key=lambda x: abs(x[1]['skew']))
                best_skew = abs(sorted_transformations[0][1]['skew'])

                # Get transformations with skewness within a margin of the best one
                candidates = [
                    (name, trans) for name, trans in sorted_transformations
                    if abs(trans['skew']) <= best_skew + margin
                ]

                # Randomly choose one of the best candidates to diversify transformations
                chosen_transform = random.choice(candidates)
                best_transform_name, best_transform = chosen_transform

                # Apply the chosen transformation
                self.df[col] = best_transform['transformed']
                
                if abs(best_transform['skew']) > skew_threshold:
                    print(f"Applied {best_transform_name} transform on '{col}', but skewness remains {best_transform['skew']:.2f}, which is higher than the desired threshold.")
                else:
                    print(f"Applied {best_transform_name} transform on '{col}' (new skewness: {best_transform['skew']:.2f})")
            else:
                print(f"No transformation could be applied successfully to '{col}'. Skewness remains {original_skew:.2f}.")

    def remove_outliers(self, columns, method='iqr', threshold=1.5):
        """Remove outliers from specified columns using IQR or Z-score method."""
        for col in columns:
            if method == 'iqr':
                # Calculate IQR
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # Remove rows outside of IQR bounds
                before_count = self.df.shape[0]
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                after_count = self.df.shape[0]
                print(f"\nRemoved {before_count - after_count} outliers from '{col}' using IQR method.")
            
            elif method == 'zscore':
                # Calculate Z-scores and filter based on threshold
                z_scores = zscore(self.df[col])
                abs_z_scores = np.abs(z_scores)
                before_count = self.df.shape[0]
                self.df = self.df[abs_z_scores < threshold]
                after_count = self.df.shape[0]
                print(f"\nRemoved {before_count - after_count} outliers from '{col}' using Z-score method.")
            else:
                print(f"\nUnknown method '{method}'. Supported methods are 'iqr' and 'zscore'.")

    def identify_highly_correlated_columns(self, threshold=0.9):
        """Step 2: Identify highly correlated columns above a certain threshold."""
        correlation_matrix = self.df.corr(numeric_only=True)
        corr_pairs = correlation_matrix.abs().unstack()
        high_corr_pairs = corr_pairs[(corr_pairs > threshold) & (corr_pairs < 1)].sort_values(ascending=False)
        
        columns_to_drop = set()
        for i in range(len(high_corr_pairs)):
            col1, col2 = high_corr_pairs.index[i]
            if col1 not in columns_to_drop and col2 not in columns_to_drop:
                columns_to_drop.add(col2)
        
        return list(columns_to_drop)

    def make_removal_decision(self, correlated_columns):
        """Step 3: Make a decision on which columns to drop."""
        print("Columns to consider dropping due to high correlation:", correlated_columns)

    def remove_highly_correlated_columns(self, threshold=0.9):
        """Step 4: Remove highly correlated columns from the dataset."""
        columns_to_drop = self.identify_highly_correlated_columns(threshold)
        self.df = self.df.drop(columns=list(columns_to_drop))
        return self.df