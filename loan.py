import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
st.set_page_config(page_title="Loan Prediction", layout="centered")

# Custom CSS
st.markdown("""
<style>
/* Background */
.stApp {
    background-color: #dce9f7;
}

/* Title */
h1 {
    text-align: center;
    color: #2c3e50;
}

/* Card-like container */
.block-container {
    padding: 2rem;
    border-radius: 10px;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}

/* Input labels */
label {
    font-weight: 500;
    color: #333;
}

/* Button */
.stButton>button {
    background-color: #2c3e50;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #1a242f;
}

/* Success & Error boxes */
.stAlert {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
# ---------------------------
# Load Dataset
# ---------------------------
data = pd.read_csv("loan_data.csv")

# Clean column names & values
data.columns = data.columns.str.strip()
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# ---------------------------
# Preprocessing
# ---------------------------
data = data.drop('loan_id', axis=1)

data['education'] = data['education'].map({'Graduate':1, 'Not Graduate':0})
data['self_employed'] = data['self_employed'].map({'Yes':1, 'No':0})
data['loan_status'] = data['loan_status'].map({'Approved':1, 'Rejected':0})

# Fill missing values
num_cols = data.select_dtypes(include=['int64','float64']).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# ---------------------------
# Train Model
# ---------------------------
X = data.drop('loan_status', axis=1)
y = data['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# UI
# ---------------------------
st.title("Loan Approval Prediction")

st.write("Enter applicant details:")

col1, col2 = st.columns(2)

with col1:
    dependents = st.number_input("No of Dependents", min_value=0, value=0)
    education = st.selectbox("Education", ["Graduate","Not Graduate"])
    income_monthly = st.number_input("Monthly Income", value=0)
    income = income_monthly * 12
    loan_amt = st.number_input("Loan Amount", min_value=0, value=0)
    cibil = st.number_input("CIBIL Score", min_value=0, value=0)

with col2:
    self_emp = st.selectbox("Self Employed", ["Yes","No"])
    loan_term = st.number_input("Loan Term (Months)", min_value=0, value=0)
    res_asset = st.number_input("Residential Assets Value", min_value=0, value=0)
    com_asset = st.number_input("Commercial Assets Value", min_value=0, value=0)
    bank_asset = st.number_input("Bank Asset Value", min_value=0, value=0)

lux_asset = st.number_input("Luxury Assets Value", min_value=0, value=0)

# ---------------------------
# Prediction with Validation
# ---------------------------
if st.button("Predict Loan Status"):

    missing_fields = []

    if income == 0:
        missing_fields.append("Annual Income")
    if loan_amt == 0:
        missing_fields.append("Loan Amount")
    if loan_term == 0:
        missing_fields.append("Loan Term")
    if cibil == 0:
        missing_fields.append("CIBIL Score")

    # Show missing prompts
    if len(missing_fields) > 0:
        for field in missing_fields:
            st.warning(f"⚠️ Please enter {field}")
    else:
        # Prepare input
        input_data = pd.DataFrame({
            'no_of_dependents':[dependents],
            'education':[1 if education=="Graduate" else 0],
            'self_employed':[1 if self_emp=="Yes" else 0],
            'income_annum':[income],
            'loan_amount':[loan_amt],
            'loan_term':[loan_term],
            'cibil_score':[cibil],
            'residential_assets_value':[res_asset],
            'commercial_assets_value':[com_asset],
            'luxury_assets_value':[lux_asset],
            'bank_asset_value':[bank_asset]
        })

        result = model.predict(input_data)[0]

        if result == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

            # Reasons
            reasons = []

            if cibil < 600:
                reasons.append("Low CIBIL Score")

            if income < 200000:
                reasons.append("Low Income")

            if loan_amt > income * 3:
                reasons.append("Loan Amount too high compared to Income")

            emi = loan_amt / loan_term

            if emi > income * 0.4:
                reasons.append("EMI is high compared to income")

            if dependents > 3:
                reasons.append("Too many dependents")

            if bank_asset < 50000:
                reasons.append("Low Financial Assets")

            if len(reasons) > 0:
                st.write("Possible Reasons:")
                for r in reasons:
                    st.write("- " + r)