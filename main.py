import streamlit as st
from prediction_helper import predict

st.title('Credit Risk Modeling')

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

with row1[0]:
    age = st.number_input('Age', min_value=18, max_value=100, step=1)
with row1[1]:
    income = st.number_input('Income', min_value=0, value=1200000)
with row1[2]:
    loan_amount = st.number_input('Loan Amount', min_value=100000, value=2560000)

loan_to_income = loan_amount / income if income > 0 else 0

with row2[0]:
    st.text("Loan to Income Ratio:")
    st.text(f"{loan_to_income:.2f}")

with row2[1]:
    loan_tenure = st.number_input('Loan Tenure (Months)', value=36)

with row2[2]:
    avg_dpd = st.number_input('Avg DPD', value=6)

with row3[0]:
    delinquency_ratio = st.number_input('Delinquency Ratio', min_value=0, max_value=100, step=1, value=20)
with row3[1]:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0, max_value=100, step=1, value=45)
with row3[2]:
    num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=3)

with row4[0]:
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])
with row4[1]:
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
with row4[2]:
    loan_type = st.selectbox('Loan Type', ['Secured', 'Unsecured'])



if st.button('Calculate Risk'):
    probability, credit_score, rating = predict(age, income, loan_amount, loan_to_income, loan_tenure, avg_dpd,
                                                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                                residence_type, loan_purpose, loan_type)

    st.write(f"Default Probability: {probability:.2f}%")
    st.write(f"Credit Score: {credit_score}")
    st.write(f"Rating: {rating}")