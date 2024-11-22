from multiprocessing.connection import default_family

import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

MODEL_PATH = 'artifacts/model_data.joblib'
model_data = joblib.load(MODEL_PATH)

model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']

def prepare_df(age, income, loan_amount, loan_to_income, loan_tenure, avg_dpd,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):
    input_data = {
        'age': [age],
        'loan_tenure_months': [loan_tenure],
        'number_of_open_accounts': [num_open_accounts],
        'credit_utilization_ratio': [credit_utilization_ratio],
        'loan_to_income': [loan_to_income],
        'delinquency_ratio': [delinquency_ratio],
        'avg_dpd_per_delinquency': [avg_dpd],
        'residence_type_Owned': [1 if residence_type == 'Owned' else 0],
        'residence_type_Rented': [1 if residence_type == 'Rented' else 0],
        'loan_purpose_Education': [1 if loan_purpose == 'Education' else 0],
        'loan_purpose_Home': [1 if loan_purpose == 'Home' else 0],
        'loan_purpose_Personal': [1 if loan_purpose == 'Personal' else 0],
        'loan_type_Unsecured': [1 if loan_type == 'Unsecured' else 0]

        # additional fields (dummy)

    }
    # our scaler has more columns than required because we dropped some columns after scaling so we
    # add dummy columns, they will not have any impact, we will remove them after scaling
    # Create DataFrame
    # Add dummy columns for extra features not provided in input
    extra_columns = [
        'number_of_dependants', 'years_at_current_address', 'zipcode',
        'sanction_amount', 'processing_fee', 'gst', 'net_disbursement',
        'principal_outstanding', 'bank_balance_at_application',
        'number_of_closed_accounts', 'enquiry_count'
    ]

    # Add extra columns with default values (e.g., 0 for numeric data)
    for col in extra_columns:
        input_data[col] = [1]  # Default value

    # Create the DataFrame
    input_df = pd.DataFrame(input_data)

    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    input_df = input_df[features]

    return input_df


def calc_credit_score(input_df, base_score=300, scale_length=600):
    # we have
    # model.coef_ and model.intercept_
    # we will use them to calc the sigmoid value

    # y = m1x1 + m2x2 + .... + b
    y = np.dot(input_df.values, model.coef_.T) + model.intercept_

    default_probability = 1 / (1 + np.exp(-y))
    non_default_probability = 1 - default_probability

    # use flatten() to get a scalar value
    credit_score = base_score + non_default_probability.flatten() * scale_length

    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'

    rating = get_rating(credit_score)

    return default_probability.flatten()[0] * 100, int(credit_score), rating

def predict(age, income, loan_amount, loan_to_income, loan_tenure, avg_dpd,
                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                residence_type, loan_purpose, loan_type):

    input_df = prepare_df(age, income, loan_amount, loan_to_income, loan_tenure, avg_dpd,
                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                residence_type, loan_purpose, loan_type)

    probability, credit_score, rating = calc_credit_score(input_df)

    return probability, credit_score, rating

