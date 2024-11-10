### Data Cleaning
In this step, we clean the dataset by handling missing values, duplicates, and ensuring data consistency.
- **Handling Missing Values**: We use the `.fillna()` or `.dropna()` methods to handle missing values.
- **Removing Duplicates**: Duplicate rows are removed to prevent bias in the analysis.

### Data Transformation
Here, we are applying transformations to make the data more suitable for analysis:
- **Normalizing Columns**: Scaling numerical values so they fall within a specific range.
- **Encoding Categorical Data**: Converting non-numerical labels into numerical form using label encoding or one-hot encoding.

### Visualization: Box and Whisker Plots
- **Box Plot Theory**: A box plot shows the distribution of data based on a five-number summary: minimum, first quartile (Q1), median, third quartile (Q3), and maximum.
- **Outliers**: Points plotted outside the whiskers represent outliers, which are calculated using the Interquartile Range (IQR). The IQR is calculated as Q3 - Q1.
- **Interpretation**: By visualizing the loan amount distribution, we can understand how most loan values are distributed, whether there are skewed values, and where outliers lie.

### Creating a Pairplot with Seaborn
- **Pairplot**: A pairplot allows us to visualize pairwise relationships between features in the dataset.
- **Use Case**: We use this to understand potential correlations between different variables like 'loan_amount' and 'interest_rate'.
- **Interpretation**: Strong linear patterns may indicate relationships between features that could be useful for predicting loan risk.

### Feature Engineering
Feature engineering involves creating new features or modifying existing ones to improve model performance. Here:
- **Creating Debt-to-Income Ratio**: We derive a new feature, the Debt-to-Income ratio, which helps determine borrower risk.

### Statistical Analysis: Correlation Matrix
- **Purpose**: Calculating the correlation matrix helps us identify which features are strongly related to each other. This is useful to reduce multicollinearity when building machine learning models.

### Machine Learning Model Training
- **Model Choice**: Here, we are using a Decision Tree Classifier to predict loan approval. Decision trees are chosen for their simplicity and interpretability.
- **Evaluation Metrics**: We use accuracy and precision metrics to evaluate the model's performance.

## Summarising results

### Key Insights
- **Loan Default Trends**: We observed that higher loan amounts with a combination of low annual incomes tend to have a higher risk of default.
- **Interest Rate Influence**: Loans with higher interest rates also seem to have a higher likelihood of late payments or defaults.

### Conclusion
The exploratory data analysis provided crucial insights into loan repayment behavior. By understanding the relationships between different variables, financial institutions can make better-informed decisions on loan approvals and risk management.

### Future Work
- **Feature Improvement**: Additional features such as credit history length and alternative credit scores could be incorporated for deeper analysis.
- **Advanced Models**: Apply more sophisticated models, such as Random Forests or XGBoost, to improve prediction accuracy.