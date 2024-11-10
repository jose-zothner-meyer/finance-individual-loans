
---

# Code Breakdown and Explanation: `DataFrameInfo` Class

This document explains each line of the code in the `DataFrameInfo` class, describing its purpose and functionality in a concise way. The `DataFrameInfo` class provides various methods to explore and summarize a pandas DataFrame.

### Code

```python
import pandas as pd
```
- **Imports the pandas library** as `pd`, a common alias that allows for easier use of pandas functions and data structures.

```python
class DataFrameInfo:
```
- **Defines a class named `DataFrameInfo`** that will contain multiple methods to operate on a DataFrame.

```python
    def __init__(self, df):
        self.df = df
```
- **Initializes the class with a DataFrame**. The `__init__` method is called when an instance of the class is created. The DataFrame (`df`) is assigned to the instance variable `self.df`.

```python
    def describe_columns(self):
        """Describe all columns in the DataFrame to check data types and non-null counts."""
        # Directly call info() without assigning to a variable
        self.df.info()
```
- **Defines the `describe_columns` method** to describe the columns in the DataFrame.
  - `self.df.info()` **displays information about the DataFrame**, including data types, non-null counts, and memory usage.

```python
    def get_statistics(self):
        """Extract statistical values: mean, median, and standard deviation."""
        # Select only numeric columns for statistical summary
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
```
- **Defines the `get_statistics` method** to extract statistical information (mean, median, and standard deviation).
  - `numeric_df = self.df.select_dtypes(include=['float64', 'int64'])` selects only numeric columns of type `float64` or `int64`.

```python
        stats = numeric_df.describe().T[['mean', '50%', 'std']]
```
- **Creates a statistical summary** by calling `describe()` on numeric columns. The `.T` transposes the DataFrame, and `[['mean', '50%', 'std']]` selects the mean, median (50%), and standard deviation.

```python
        stats.rename(columns={'50%': 'median'}, inplace=True)
```
- **Renames the column labeled '50%' to 'median'** for better clarity.

```python
        return stats
```
- **Returns the `stats` DataFrame** containing the mean, median, and standard deviation for further analysis.

```python
    def count_distinct_values(self, columns):
        """Count distinct values in specified categorical columns."""
        distinct_counts = {col: self.df[col].nunique() for col in columns}
```
- **Defines the `count_distinct_values` method** to count distinct values in specified columns.
  - `{col: self.df[col].nunique() for col in columns}` is a dictionary comprehension that calculates the number of unique values (`nunique()`) for each specified column.

```python
        return distinct_counts
```
- **Returns the dictionary** of distinct value counts for each specified column.

```python
    def get_shape(self):
        """Return the shape of the DataFrame."""
        shape = self.df.shape
        return shape
```
- **Defines the `get_shape` method** to return the shape of the DataFrame.
  - `self.df.shape` returns a tuple representing the number of rows and columns in the DataFrame.

```python
    def count_nulls(self):
        """Generate a count and percentage of NULL values in each column."""
        null_counts = self.df.isnull().sum()
```
- **Defines the `count_nulls` method** to compute the count and percentage of null values in each column.
  - `null_counts = self.df.isnull().sum()` calculates the total number of null values for each column.

```python
        null_percent = (null_counts / len(self.df)) * 100
```
- **Calculates the percentage of null values** for each column based on the length of the DataFrame.

```python
        null_summary = pd.DataFrame({
            'Null Count': null_counts,
            'Percentage': null_percent
        })
```
- **Creates a DataFrame (`null_summary`)** with two columns: `Null Count` and `Percentage` to summarize the null values.

```python
        return null_summary
```
- **Returns the `null_summary` DataFrame** for further analysis.

```python
    def summarize_info(self):
        """Summarize all information of the DataFrame for a quick overview."""
        # Collect all data without printing, for better use in interactive environments
        self.describe_columns()
```
- **Defines the `summarize_info` method** to summarize information about the DataFrame.
  - Calls `self.describe_columns()` to provide an overview of the DataFrame columns (prints column data types, counts, etc.).

```python
        stats = self.get_statistics()
```
- **Calls `get_statistics()`** and stores the result in `stats` to collect summary statistics like mean, median, and standard deviation.

```python
        distinct_counts = self.count_distinct_values(self.df.select_dtypes(include=['category', 'object']).columns)
```
- **Calculates distinct values** for all categorical or object columns using `count_distinct_values()`.

```python
        null_summary = self.count_nulls()
```
- **Calls `count_nulls()`** to generate a summary of null values in the DataFrame.

```python
        shape = self.get_shape()
```
- **Calls `get_shape()`** to get the number of rows and columns in the DataFrame.

```python
        return {
            "statistics": stats,
            "distinct_counts": distinct_counts,
            "null_summary": null_summary,
            "shape": shape
        }
```
- **Returns a dictionary containing** `statistics`, `distinct_counts`, `null_summary`, and `shape` for further analysis or interactive use.

---

### Summary
- The `DataFrameInfo` class is a utility for **analyzing and summarizing a pandas DataFrame**.
- Each method serves a distinct purpose, such as **describing columns, summarizing statistics, counting distinct values, or generating null value summaries**.
- This class can be particularly useful in interactive environments where users need a quick overview of their data without printing outputs directly.