
---

# Code Breakdown and Explanation: `DataTransform` Class

The `DataTransform` class is designed to handle various data transformation operations on a pandas DataFrame, with logging functionality to track progress and catch errors. Below, each line of the code is explained to clarify its purpose.

### Code

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PowerTransformer
import logging
import re
from typing import List, Tuple, Optional, Dict
```
- **Import necessary libraries**:
  - `pandas` for data manipulation.
  - `LabelEncoder` and `PowerTransformer` for encoding categorical features and transformations.
  - `logging` for recording transformation activities and errors.
  - `re` for regular expressions, used in text processing.
  - `typing` for type hints to make the code more readable and easier to debug.

```python
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='data_transform.log', filemode='a')
```
- **Configures logging**:
  - Logs are saved to a file named `data_transform.log`.
  - Log level is set to `INFO`, which records informational messages and warnings.

```python
class DataTransform:
```
- **Defines the `DataTransform` class** for transforming a DataFrame, including multiple methods for handling null values, encoding, and type conversions.

```python
    def __init__(self, df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df
        self.label_encoders: Dict[str, LabelEncoder] = {}  # Dictionary to store the label encoders
        logging.info("Initialized DataTransform class.")
```
- **Initializes the class with a DataFrame** and a dictionary (`label_encoders`) to store label encoders used in label encoding transformations.

```python
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
```
- **Converts specified columns to datetime** format. If some values cannot be converted, they are set to `NaT` and logged as warnings.

```python
    def convert_to_numerical(self, column: str, remove_text: Optional[str] = None) -> None:
        """Convert specified column to numeric by removing text if specified."""
        try:
            if remove_text:
                self.df[column] = self.df[column].astype(str).str.replace(remove_text, '', regex=True).astype(float)
            else:
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
            missing_count: int = self.df[column].isna().sum()
            if missing_count > 0:
                logging.warning(f"{missing_count} values could not be converted in column '{column}'.")
            logging.info(f"Converted column '{column}' to numerical.")
        except Exception as e:
            logging.error(f"Error converting column {column} to numerical: {e}")
```
- **Converts a column to numeric values**. If text needs to be removed, it uses a regular expression. Any conversion errors are logged.

```python
    def set_categorical(self, columns: List[str]) -> None:
        """Convert specified columns to categorical data type."""
        for column in columns:
            try:
                self.df[column] = self.df[column].astype('category')
                logging.info(f"Set column '{column}' to categorical.")
            except Exception as e:
                logging.error(f"Error converting column {column} to categorical: {e}")
```
- **Sets the specified columns to categorical data type**, making them more memory efficient and suitable for certain analysis methods.

```python
    def convert_employment_length(self, column: str) -> None:
        """Convert employment_length to numerical format."""
        def transform_employment_length(value: str) -> Optional[float]:
            if pd.isna(value):
                return None
            elif '<1' in value:
                return 0.5
            elif '10+' in value:
                return 15.0
            else:
                match = re.search(r'\d+', value)
                if match:
                    return float(match.group())
                else:
                    return None

        self.df[column] = self.df[column].astype(str).apply(transform_employment_length).astype(float)
```
- **Converts employment length information to numerical format** using a custom function. For example, `<1` is converted to `0.5`, `10+` to `15.0`, and other numbers are extracted from strings.

```python
    def convert_to_int64(self, columns: List[str]) -> None:
        """Convert specified columns to int64 data type, handling NaN values appropriately."""
        for column in columns:
            try:
                if self.df[column].isna().any():
                    self.df[column].fillna(0, inplace=True)
                    logging.info(f"Filled NaN values with 0 in column '{column}'.")
                self.df[column] = self.df[column].astype('int64')
                logging.info(f"Converted column '{column}' to int64.")
            except Exception as e:
                logging.error(f"Error converting column {column} to int64: {e}")
```
- **Converts specified columns to `int64` data type** after handling missing values by replacing them with `0`.

```python
    def summarize_transforms(self) -> None:
        """Provide a summary of the transformed DataFrame."""
        try:
            info = self.df.info()
            logging.info("DataFrame info:\n" + str(info))
            logging.info("Sample Data:\n" + str(self.df.head()))
        except Exception as e:
            logging.error(f"Error summarizing transforms: {e}")
```
- **Provides a summary of the DataFrame** after transformation, including data types and a sample of the data.

```python
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
```
- **Provides descriptive statistics** for numerical and categorical columns in the DataFrame.

```python
    def one_hot_encode_columns(self, columns: List[str]) -> None:
        """Apply One-Hot Encoding to specified columns and print the added columns."""
        for col in columns:
            if col in self.df.columns:
                original_columns = set(self.df.columns)
                self.df = pd.get_dummies(self.df, columns=[col], drop_first=True)
                new_columns = set(self.df.columns) - original_columns
                new_columns_list: List[str] = list(new_columns)
                if new_columns_list:
                    logging.info(f"Applied One-Hot Encoding to column '{col}', new columns added: {new_columns_list}")
                    print(f"One-Hot Encoding applied to column '{col}', new columns added: {new_columns_list}")
```
- **Applies One-Hot Encoding** to specified columns, dropping the first level to avoid multicollinearity. The new columns added are logged.

```python
    def transform(self) -> pd.DataFrame:
        """Apply Label and One-Hot Encoding to the DataFrame."""
        label_columns: List[str] = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'purpose']
        one_hot_columns: List[str] = ['payment_plan', 'application_type']

        self.label_encode_columns(label_columns)
        self.one_hot_encode_columns(one_hot_columns)

        return self.df
```
- **Applies label encoding and one-hot encoding** to specified columns and returns the transformed DataFrame.

```python
    def label_encode_columns(self, columns: List[str]) -> None:
        """Apply Label Encoding to specific columns and replace original values."""
        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                logging.info(f"Applied Label Encoding to column '{col}'.")
```
- **Applies label encoding to specified columns** and stores the encoders for each column, enabling future inverse transformations if needed.

```python
    def get_label_mappings(self) -> Dict[str, Dict[str, int]]:
        """Retrieve mappings of each label encoder."""
        mappings: Dict[str, Dict[str, int]] = {}
        for col, encoder in self.label_encoders.items():
            mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        return mappings
```
- **Provides a mapping of label encodings** for each column, showing the conversion between categorical values and their numerical labels.

---

### Summary
- The `DataTransform` class is equipped to handle **various data transformations** on a pandas DataFrame, such as:
  - **Converting columns to datetime, numeric, or categorical types.**
  - **Handling missing values** and **encoding categorical variables**.
  - **Applying label and one-hot encoding** with the capability to log all actions.
  - Providing detailed **logging to track operations**, which is helpful in debugging and understanding each transformation step.