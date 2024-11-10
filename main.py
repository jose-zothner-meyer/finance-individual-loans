# Import the necessary functions and classes from db_utils
from Others.aa_db_utils import load_credentials, RDSDatabaseConnector, save_to_csv

# Main script execution
if __name__ == "__main__":
    # Load credentials from the YAML file
    credentials = load_credentials()

    # Initialize the database connector with the credentials
    db_connector = RDSDatabaseConnector(credentials)

    # Extract data from the 'loan_payments' table (or any other table name you pass)
    data = db_connector.extract_data()

    # Save the extracted data to a CSV file
    save_to_csv(data, "b_df_1_loan_payments.csv")