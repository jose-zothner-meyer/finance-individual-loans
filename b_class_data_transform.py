import pandas as pd
from sklearn.preprocessing import LabelEncoder, PowerTransformer
import logging
import re
from typing import List, Tuple, Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='data_transform.log', filemode='a')

# Implementing the DataTransform class with logging
class DataTransform:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df
        self.label_encoders: Dict[str, LabelEncoder] = {}  # Dictionary to store the label encoders
        logging.info("Initialized DataTransform class.")

    def convert_to_datetime(self, columns: List[str], date_format: str = "%b-%Y") -> None:
        """Convert specified columns to datetime. Default format is '%b-%Y' (e.g., 'Jan-2021')."""
        for column in columns:
            try:
                self.df[column] = pd.to_datetime(self.df[column], format=date_format, errors='coerce')
                missing_count: int = self.df[column].isna().sum()
                if missing_count > 0:
                    logging.warning(f"{missing_count} values could not be converted in column '{column}'.")
                logging.info(f"Successfully converted column '{column}' to datetime.")
            except Exception as e:
                logging.error(f"Error converting column {column} to datetime: {e}")
    
    def convert_to_numerical(self, column: str, remove_text: Optional[str] = None) -> None:
        """Convert specified column to numeric by removing text if specified."""
        try:
            if remove_text:
                # Ensure the column is treated as a string before using `.str`
                self.df[column] = self.df[column].astype(str).str.replace(remove_text, '', regex=True).astype(float)
            else:
                # Directly convert to numeric if no text removal is needed
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
            missing_count: int = self.df[column].isna().sum()
            if missing_count > 0:
                logging.warning(f"{missing_count} values could not be converted in column '{column}'.")
            logging.info(f"Converted column '{column}' to numerical.")
        except Exception as e:
            logging.error(f"Error converting column {column} to numerical: {e}")
        
    def set_categorical(self, columns: List[str]) -> None:
        """Convert specified columns to categorical data type."""
        for column in columns:
            try:
                self.df[column] = self.df[column].astype('category')
                logging.info(f"Set column '{column}' to categorical.")
            except Exception as e:
                logging.error(f"Error converting column {column} to categorical: {e}")

    def convert_employment_length(self, column: str) -> None:
        """Convert employment_length to numerical format."""
        # Define a function to perform the transformation
        def transform_employment_length(value: str) -> Optional[float]:
            if pd.isna(value):
                return None  # Keep NaN values as-is
            elif '<1' in value:
                return 0.5
            elif '10+' in value:
                return 15.0
            else:
                # Extract numeric part and convert it to float
                match = re.search(r'\d+', value)
                if match:
                    return float(match.group())
                else:
                    return None

        # Apply the transformation
        self.df[column] = self.df[column].astype(str).apply(transform_employment_length).astype(float)

    def convert_to_int64(self, columns: List[str]) -> None:
        """Convert specified columns to int64 data type, handling NaN values appropriately."""
        for column in columns:
            try:
                # Handle NaN values before conversion: replace with 0, or drop, based on your requirement
                if self.df[column].isna().any():
                    self.df[column].fillna(0, inplace=True)
                    logging.info(f"Filled NaN values with 0 in column '{column}'.")

                # Convert to int64
                self.df[column] = self.df[column].astype('int64')
                logging.info(f"Converted column '{column}' to int64.")
            except Exception as e:
                logging.error(f"Error converting column {column} to int64: {e}")
    
    def summarize_transforms(self) -> None:
        """Provide a summary of the transformed DataFrame."""
        try:
            info = self.df.info()
            logging.info("DataFrame info:\n" + str(info))
            logging.info("Sample Data:\n" + str(self.df.head()))
        except Exception as e:
            logging.error(f"Error summarizing transforms: {e}")
        
    def summary_statistics(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Provide summary statistics of numerical and categorical columns."""
        try:
            numerical_summary: pd.DataFrame = self.df.describe()
            categorical_summary: pd.DataFrame = self.df.describe(include=['category'])
            logging.info("Numerical Columns Summary:\n" + str(numerical_summary))
            logging.info("Categorical Columns Summary:\n" + str(categorical_summary))
            return numerical_summary, categorical_summary
        except Exception as e:
            logging.error(f"Error providing summary statistics: {e}")
            return None, None

    def one_hot_encode_columns(self, columns: List[str]) -> None:
        """Apply One-Hot Encoding to specified columns and print the added columns."""
        for col in columns:
            if col in self.df.columns:
                # Capture original column names before transformation
                original_columns = set(self.df.columns)

                # Apply One-Hot Encoding using pandas.get_dummies()
                self.df = pd.get_dummies(self.df, columns=[col], drop_first=True)

                # Capture new columns after One-Hot Encoding
                new_columns = set(self.df.columns) - original_columns

                # Log and print the new columns added as a result of One-Hot Encoding
                new_columns_list: List[str] = list(new_columns)
                if new_columns_list:
                    logging.info(f"Applied One-Hot Encoding to column '{col}', new columns added: {new_columns_list}")
                    print(f"One-Hot Encoding applied to column '{col}', new columns added: {new_columns_list}")

    def transform(self) -> pd.DataFrame:
        """Apply Label and One-Hot Encoding to the DataFrame."""
        label_columns: List[str] = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'purpose']
        one_hot_columns: List[str] = ['payment_plan', 'application_type']

        # Apply Label Encoding
        self.label_encode_columns(label_columns)

        # Apply One-Hot Encoding
        self.one_hot_encode_columns(one_hot_columns)

        return self.df

    def label_encode_columns(self, columns: List[str]) -> None:
        """Apply Label Encoding to specific columns and replace original values."""
        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le  # Store the LabelEncoder for possible inverse_transform
                logging.info(f"Applied Label Encoding to column '{col}'.")
    
    def get_label_mappings(self) -> Dict[str, Dict[str, int]]:
        """Retrieve mappings of each label encoder."""
        mappings: Dict[str, Dict[str, int]] = {}
        for col, encoder in self.label_encoders.items():
            mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        return mappings