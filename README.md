# Financial Institution Loan Portfolio EDA
## UNDER DEVELOPMENT. 
Most recent changes are extensions of feature engineering (a_notebook_5.ipynb) and Label- and One-Hot Encodings that can be found (a_notebook_1.ipynb). Still need to work on the processes and logic of the alteration steps in order to avoid missleads.

## Overview
A comprehensive analysis of a financial institution's loan portfolio using Python, focusing on data preprocessing, exploratory data analysis (EDA), and feature engineering. The project aims to derive insights for loan approvals, risk assessment, and portfolio management.

## Key Features
- Data cleaning and preprocessing
- Advanced feature engineering
- Correlation analysis
- Principal Component Analysis (PCA)
- Feature importance ranking
- Statistical analysis and visualization
- Outlier detection and handling

## Technical Implementation
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling
- **Feature Engineering**: Created new features and transformed existing ones for better model performance
- **Dimensionality Reduction**: Implemented PCA for feature selection
- **Statistical Analysis**: Correlation analysis and feature importance ranking using Random Forest
- **Visualization**: Comprehensive plots using seaborn and matplotlib

## Project Structure
```
├── notebooks/
│   ├── a_notebook_1.ipynb (Data Loading & Preprocessing)
│   ├── a_notebook_2.ipynb (Feature Engineering)
│   ├── a_notebook_3.ipynb (Statistical Analysis)
│   ├── a_notebook_4.ipynb (Feature Selection & PCA)
│   └── a_notebook_5.ipynb (Advanced financial feature engineering)
├── classes/
│   ├── b_class_data_frame_info.py
│   ├── b_class_data_frame_transform.py
│   ├── b_class_data_transform.py
│   ├── b_class_feature_reduction.py
│   └── b_class_plotter.py
├── LICENSE/
├── main.py
├── README.md
├── REQUIREMENTS.txt
└── text.md
```
## Detailed File Structure

### Notebooks
- **a_notebook_1.ipynb**: Initial data loading, cleaning, and categorical encoding implementation
- **a_notebook_2.ipynb**: Feature engineering, outlier detection, and handling missing values
- **a_notebook_3.ipynb**: Statistical analysis, correlation studies, and distribution analysis
- **a_notebook_4.ipynb**: Feature selection, PCA implementation, and importance ranking
- **a_notebook_5.ipynb**: Advanced feature engineering and model preparation, including:
  - Feature interaction analysis
  - Polynomial feature creation
  - Time-based feature engineering
  - Advanced categorical encoding
  - Feature scaling and standardization
  - Final dataset preparation for modeling

### Python Classes
- **b_class_data_frame_transform.py**: Core class containing methods for:
  - Data type conversions
  - Feature scaling and normalization
  - Encoding categorical variables
  - Handling missing values
  - Outlier detection and removal

- **b_class_plotter.py**: Visualization class with methods for:
  - Correlation heatmaps
  - Distribution plots
  - Feature importance charts
  - PCA visualization
  - Statistical analysis plots

- **b_class_feature_reduction.py**: Feature engineering class implementing:
  - PCA transformation
  - Feature selection algorithms
  - Dimensionality reduction
  - Variance analysis
  - Feature importance calculation

- **b_class_data_frame_info.py**: Utility class providing:
  - DataFrame information summaries
  - Data quality checks
  - Column statistics
  - Missing value reports
  - Data type management

### Configuration Files
- **requirements.txt**: List of Python dependencies
- **README.md**: Project documentation and setup instructions
- **.gitignore**: Git ignore rules for the project

### Data (Not Public)
- **raw_loan_data.csv**: Original loan dataset
- **processed_loan_data.csv**: Cleaned and preprocessed dataset
- **transformed_loan_data.csv**: Final dataset with engineered features

Note: Data files are not included in the repository for privacy reasons. Please contact the repository owner for data access requests.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jose-zothner-meyer/finance-customer-loans.git
cd finance-customer-loans
```

2. Create, install required packages and activate conda environment:
```bash
conda create --name finance_loans --file requirements.txt
conda activate finance_loans
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Navigate through the notebooks in sequential order (1-5)

## Data Access
The loan dataset used in this project is not publicly available due to privacy concerns. For access to the dataset or questions about the analysis, please contact me at:
- Email: zothner.meyer.jose@gmail.com
- LinkedIn: [Jose Zothner Meyer](https://www.linkedin.com/in/josemeyer)

### Full loans dataset schema

- **id**: unique id of the loan
- **member_id**: id of the member to took out the loan
- **loan_amount**: amount of loan the applicant received
- **funded_amount**: The total amount committed to the loan at the point in time 
- **funded_amount_inv**: The total amount committed by investors for that loan at that point in time 
- **term**: The number of monthly payments for the loan
- **int_rate**: Interest rate on the loan
- **instalment**: The monthly payment owned by the borrower
- **grade**: LC assigned loan grade
- **sub_grade**: LC assigned loan sub grade
- **employment_length**: Employment length in years.
- **home_ownership**: The home ownership status provided by the borrower
- **annual_inc**: The annual income of the borrower
- **verification_status**: Indicates whether the borrowers income was verified by the LC or the income source was verified
- **issue_date:** Issue date of the loan
- **loan_status**: Current status of the loan
- **payment_plan**: Indicates if a payment plan is in place for the loan. Indication borrower is struggling to pay.
- **purpose**: A category provided by the borrower for the loan request.
- **dti**: A ratio calculated using the borrowerâ€™s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrowerâ€™s self-reported monthly income.
- **delinq_2yr**: The number of 30+ days past-due payment in the borrower's credit file for the past 2 years.
- **earliest_credit_line**: The month the borrower's earliest reported credit line was opened
- **inq_last_6mths**: The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
- **mths_since_last_record**: The number of months since the last public record.
- **open_accounts**: The number of open credit lines in the borrower's credit file.
- **total_accounts**: The total number of credit lines currently in the borrower's credit file
- **out_prncp**: Remaining outstanding principal for total amount funded
- **out_prncp_inv**: Remaining outstanding principal for portion of total amount funded by investors
- **total_payment**: Payments received to date for total amount funded
- **total_rec_int**: Interest received to date
- **total_rec_late_fee**: Late fees received to date
- **recoveries**: post charge off gross recovery
- **collection_recovery_fee**: post charge off collection fee
- **last_payment_date**: Last month payment was received
- **last_payment_amount**: Last total payment amount received
- **next_payment_date**: Next scheduled payment date
- **last_credit_pull_date**: The most recent month LC pulled credit for this loan
- **collections_12_mths_ex_med**: Number of collections in 12 months excluding medical collections
- **mths_since_last_major_derog**: Months since most recent 90-day or worse rating
- **policy_code**: publicly available policy_code=1 new products not publicly available policy_code=2
- **application_type**: Indicates whether the loan is an individual application or a joint application with two co-borrowers

## Key Findings
- Outstanding principal and payment features are the strongest predictors
- Identified key correlations between loan characteristics
- Reduced feature dimensionality while maintaining 95% variance
- Developed robust feature selection methodology

## Future Improvements
- Implementation of advanced ML models
- Time series analysis of loan performance
- Risk scoring system development
- Interactive dashboard creation

## Credits
This project was developed as part of the AiCore program. Special thanks to [AiCore](https://www.theaicore.com) for their guidance and support.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

[def]: #full-loans-dataset-schema