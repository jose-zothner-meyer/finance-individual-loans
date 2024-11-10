import pandas as pd

class DataFrameInfo:
    def __init__(self, df):
        self.df = df

    def describe_columns(self):
        """Describe all columns in the DataFrame to check data types and non-null counts."""
        # Directly call info() without assigning to a variable
        self.df.info()
    
    def get_statistics(self):
        """Extract statistical values: mean, median, and standard deviation."""
        # Select only numeric columns for statistical summary
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        stats = numeric_df.describe().T[['mean', '50%', 'std']]
        stats.rename(columns={'50%': 'median'}, inplace=True)
        
        # Return the numeric statistics without formatting for further use
        return stats
    
    def count_distinct_values(self, columns):
        """Count distinct values in specified categorical columns."""
        distinct_counts = {col: self.df[col].nunique() for col in columns}
        # Return the distinct counts dictionary for flexibility
        return distinct_counts
    
    def get_shape(self):
        """Return the shape of the DataFrame."""
        shape = self.df.shape
        return shape
    
    def count_nulls(self):
        """Generate a count and percentage of NULL values in each column."""
        null_counts = self.df.isnull().sum()
        null_percent = (null_counts / len(self.df)) * 100
        null_summary = pd.DataFrame({
            'Null Count': null_counts,
            'Percentage': null_percent
        })
        # Return the null summary DataFrame for further analysis
        return null_summary
    
    def summarize_info(self):
        """Summarize all information of the DataFrame for a quick overview."""
        # Collect all data without printing, for better use in interactive environments
        self.describe_columns()
        stats = self.get_statistics()
        distinct_counts = self.count_distinct_values(self.df.select_dtypes(include=['category', 'object']).columns)
        null_summary = self.count_nulls()
        shape = self.get_shape()
        
        # Returning values for further analysis if needed
        return {
            "statistics": stats,
            "distinct_counts": distinct_counts,
            "null_summary": null_summary,
            "shape": shape
        }