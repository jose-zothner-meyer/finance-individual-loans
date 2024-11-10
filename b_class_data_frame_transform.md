
---

# Code Breakdown and Explanation: `DataFrameTransform` Class

The `DataFrameTransform` class helps preprocess and transform a pandas DataFrame, handling issues like null values, skewness, and outliers. This explanation provides a line-by-line description of what each part of the code does.

### Code

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from scipy import stats  # For Box-Cox transformation
from scipy.stats import zscore
from b_class_data_transform import DataTransform
```
- **Imports necessary libraries** for data manipulation, transformation, and statistical analysis:
  - `pandas` and `numpy` for data manipulation.
  - `PowerTransformer` from `sklearn` for Yeo-Johnson transformations.
  - `stats` from `scipy` for statistical transformations like Box-Cox.
  - Imports a custom class `DataTransform` for label encoding and one-hot encoding.

```python
class DataFrameTransform:
```
- **Defines the `DataFrameTransform` class** to hold methods for transforming and analyzing a pandas DataFrame.

```python
    def __init__(self, df):
        self.data_transform = DataTransform(df)
        self.df = df
```
- **Initializes the class** with a DataFrame (`df`) and creates an instance of `DataTransform` to use later for categorical transformations.

```python
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
```
- **Defines `check_nulls` to count and display null values** in each column as a summary table with counts and percentages.

```python
    def drop_columns(self, threshold=50.0):
        """Drop columns with a percentage of NULL values above the threshold."""
        to_drop = self.df.columns[self.df.isnull().mean() * 100 > threshold]
        self.df.drop(columns=to_drop, inplace=True)
        print(f"\nDropped columns with more than {threshold}% missing data: {list(to_drop)}")
```
- **Drops columns where the percentage of null values exceeds a given threshold** (default is 50%).

```python
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
                    if not self.df[column].mode().empty:
                        self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                    else:
                        self.df[column].fillna("Unknown", inplace=True)
        print(f"\nImputed missing values using the {method} method for numerical and mode for categorical columns.")
```
- **Imputes missing values** in numeric columns using either the mean or median, and categorical columns using the mode.

```python
    def confirm_no_nulls(self):
        """Confirm that all NULL values have been removed."""
        null_summary = self.df.isnull().sum().sum()
        print(f"\nTotal remaining NULL values in DataFrame: {null_summary}")
        if null_summary == 0:
            print("All NULL values have been successfully removed or imputed.")
        else:
            print("There are still NULL values in the DataFrame.")
        return null_summary
```
- **Confirms that all null values have been dealt with** and prints the result.

```python
    def transform_dataframe(self):
        # Apply label encoding and one-hot encoding using the DataTransform class
        self.data_transform.transform()

        # Return the transformed DataFrame
        return self.data_transform.df
```
- **Transforms the DataFrame** using the `DataTransform` class and returns the transformed DataFrame.

```python
    def get_label_mappings(self):
        """Get the label mappings after applying label encoding."""
        return self.data_transform.get_label_mappings()
```
- **Returns the label mappings** used in categorical encoding by `DataTransform`.

```python
    def identify_skewed_columns(self, threshold=0.5):
        # Identify columns with skewness above the threshold
        skewed_cols = self.df.select_dtypes(include=['float64', 'int64']).apply(lambda x: x.skew()).sort_values(ascending=False)
        skewed_cols = skewed_cols[abs(skewed_cols) > threshold]
        return skewed_cols.index.tolist()
```
- **Identifies columns with skewness** greater than the specified threshold (default 0.5).

```python
    def transform_skewed_columns(self, columns):
        """Apply transformations to skewed columns to reduce skewness."""
        for col in columns:
            original_skew = self.df[col].skew()
            print(f"\nOriginal skewness for '{col}': {original_skew:.2f}")
            
            # Dictionary to store transformations
            transformations = {}
            ...
```
- **Applies transformations to reduce skewness** for specified columns, trying several transformations (e.g., Box-Cox, Yeo-Johnson, log, sqrt).

```python
    def log_transform(self, col):
        """Log transformation (handles non-positive values by shifting)."""
        ...
```
- **Performs log transformation**, shifting non-positive values to avoid math errors.

```python
    def sqrt_transform(self, col):
        """Square root transformation (handles non-positive values by shifting)."""
        ...
```
- **Performs square root transformation**, handling non-positive values by shifting them.

```python
    def reciprocal_transform(self, col):
        """Reciprocal transformation (handles non-zero values)."""
        ...
```
- **Performs reciprocal transformation**, adding a small value to avoid division by zero.

```python
    def boxcox_transform(self, col):
        """Box-Cox transformation (only for positive values)."""
        ...
```
- **Applies the Box-Cox transformation**, which works only for positive values.

```python
    def yeojohnson_transform(self, col):
        """Yeo-Johnson transformation (works with any range of values)."""
        ...
```
- **Uses the Yeo-Johnson transformation**, which can be applied to any range of values.

```python
    def remove_outliers(self, columns, method='iqr', threshold=1.5):
        """Remove outliers from specified columns using IQR or Z-score method."""
        for col in columns:
            if method == 'iqr':
                # Calculate IQR and remove outliers
                ...
            elif method == 'zscore':
                # Remove outliers using Z-score
                ...
            else:
                print(f"\nUnknown method '{method}'. Supported methods are 'iqr' and 'zscore'.")
```
- **Removes outliers** using either the Interquartile Range (IQR) method or the Z-score method.

```python
    def identify_highly_correlated_columns(self, threshold=0.9):
        """Identify highly correlated columns above a certain threshold."""
        correlation_matrix = self.df.corr(numeric_only=True)
        corr_pairs = correlation_matrix.abs().unstack()
        high_corr_pairs = corr_pairs[(corr_pairs > threshold) & (corr_pairs < 1)].sort_values(ascending=False)
        
        columns_to_drop = set()
        for i in range(len(high_corr_pairs)):
            col1, col2 = high_corr_pairs.index[i]
            if col1 not in columns_to_drop and col2 not in columns_to_drop:
                columns_to_drop.add(col2)
        
        return list(columns_to_drop)
```
- **Identifies highly correlated columns** whose correlation is above a given threshold and suggests removing them to reduce multicollinearity.

```python
    def make_removal_decision(self, correlated_columns):
        """Make a decision on which columns to drop."""
        print("Columns to consider dropping due to high correlation:", correlated_columns)
```
- **Prints columns with high correlation** to help the user decide which ones to remove.

```python
    def remove_highly_correlated_columns(self, threshold=0.9):
        """Remove highly correlated columns from the dataset."""
        columns_to_drop = self.identify_highly_correlated_columns(threshold)
        self.df = self.df.drop(columns=list(columns_to_drop))
        return self.df
```
- **Removes columns** with high correlation based on a specified threshold and returns the updated DataFrame.

---

### Summary
- The `DataFrameTransform` class provides comprehensive methods for **handling null values, encoding, outlier removal, and dealing with skewed data**.
- Each method is tailored to prepare data for further analysis or modeling by reducing bias, skewness, and multicollinearity, as well as transforming categorical values.
- This class can significantly simplify data preprocessing for machine learning and statistical analysis.