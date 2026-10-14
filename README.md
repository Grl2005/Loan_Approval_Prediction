# Loan Approval Prediction System

An interactive machine learning web application that predicts whether a loan application is likely to be approved or rejected based on an applicant's financial and personal details.

## Features

- **Loan Approval Prediction:** Predicts loan approval status using a Random Forest Classifier.
- **Applicant Information:** Accepts income, loan amount, CIBIL score, education, employment status, dependents, and asset values.
- **Input Validation:** Checks mandatory fields before generating a prediction.
- **Rejection Reason Analysis:** Displays possible reasons for rejection based on predefined financial conditions.
- **Interactive Web Interface:** Provides a user-friendly interface built using Streamlit.
- **Data Preprocessing:** Handles missing values and converts categorical values into numerical representations.

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn

## Machine Learning Algorithm

### Random Forest Classifier

The project uses a Random Forest Classifier, an ensemble machine learning algorithm that combines multiple decision trees to make a classification prediction.

The model is configured with 100 decision trees using:

`RandomForestClassifier(n_estimators=100, random_state=42)`

Each tree produces a prediction, and the classifier combines the trees' predictions to determine the final loan approval status.

## Project Workflow

1. Load historical loan application data from a CSV file.
2. Clean column names and remove unnecessary columns.
3. Handle missing numerical values using mean imputation.
4. Convert categorical values, such as education and employment status, into numerical representations.
5. Separate the input features from the target variable.
6. Split the dataset into training and testing sets using an 80:20 ratio.
7. Train the Random Forest Classifier on the training data.
8. Accept applicant details through the Streamlit interface.
9. Predict loan approval status and display possible rejection reasons.

## Dataset

The application requires a CSV dataset named `loan_data.csv` containing historical loan applicant information and loan approval status.

The dataset is not included in this repository. Obtain it from its original source, check the applicable licence and redistribution terms, and place the file in the project directory before running the application.

The expected columns include:

- `loan_id`
- `no_of_dependents`
- `education`
- `self_employed`
- `income_annum`
- `loan_amount`
- `loan_term`
- `cibil_score`
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`
- `loan_status`

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Loan-Approval-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Loan-Approval-Prediction
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the required `loan_data.csv` file in the project directory.

### 5. Run the application

```bash
streamlit run Loan.py
```

Open the local URL displayed in the terminal to use the application.

## Future Enhancements

- Compare Random Forest with other classification algorithms.
- Add confusion matrix, precision, recall, and F1-score evaluation.
- Improve the explanation of individual predictions.
- Deploy the application online.
- Improve data validation and fairness analysis.

## Disclaimer

This project is developed for educational purposes. Its predictions are not official financial decisions and should not be used as the sole basis for approving or rejecting real loan applications.
