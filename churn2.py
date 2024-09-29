import streamlit as st
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("D:\\telco_churn.csv")

# Preprocess the dataset
x = df.drop('Churn', axis=1)
y = df['Churn']

# Convert categorical variables to dummy variables
x = pd.get_dummies(x)

# Handle imbalanced dataset
samp = SMOTEENN()
x_resampled, y_resampled = samp.fit_resample(x, y)

# Split the dataset
xr_train, xr_test, yr_train, yr_test = train_test_split(x_resampled, y_resampled, test_size=0.2)

# Train the model
model2_smote = RandomForestClassifier()
model2_smote.fit(xr_train, yr_train)

# Save the model and columns
filename = 'model.sav'
pickle.dump(model2_smote, open(filename, 'wb'))
columns_filename = 'columns.sav'
pickle.dump(x.columns, open(columns_filename, 'wb'))

# Load the model and columns
load_model = pickle.load(open(filename, 'rb'))
model_columns = pickle.load(open(columns_filename, 'rb'))

gender_dict = {'Female': 0, 'Male': 1}
yes_no_dict = {'No': 0, 'Yes': 1}
internet_service_dict = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
contract_dict = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
payment_method_dict = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}

# Streamlit app
if __name__ == '__main__':
    st.title("Customer Churn Analysis")

    # Input fields
    SeniorCitizen = st.number_input("SeniorCitizen", min_value=0, max_value=1)
    MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0)
    TotalCharges = st.number_input("TotalCharges", min_value=0.0)
    gender = st.selectbox("Gender", ["Female", "Male"])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    tenure = st.number_input("Tenure", min_value=0)
    TechSupport = st.selectbox("Tech Support", ["No", "Yes"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])

    # Predict button
    predict_btt = st.button("Predict")

    if predict_btt:
        try:
            input_data = {
                'SeniorCitizen': SeniorCitizen,
                'tenure': tenure,
                'MonthlyCharges': MonthlyCharges,
                'TotalCharges': TotalCharges,
                'gender': gender,
                'Partner': Partner,
                'Dependents': Dependents,
                'InternetService': InternetService,
                'PhoneService': PhoneService,
                'MultipleLines': MultipleLines,
                'Contract': Contract,
                'PaperlessBilling': PaperlessBilling,
                'PaymentMethod': PaymentMethod,
                'TechSupport': TechSupport,
                'OnlineBackup': OnlineBackup,
                'StreamingTV': StreamingTV,
                'DeviceProtection': DeviceProtection,
                'OnlineSecurity': OnlineSecurity,
                'StreamingMovies': StreamingMovies,
            }

            # Create a DataFrame
            input_df = pd.DataFrame([input_data])

            # Convert categorical variables to dummy variables
            input_df = pd.get_dummies(input_df)

            # Ensure the input data has the same columns as the training data
            input_df = input_df.reindex(columns=model_columns, fill_value=0)

            # Make prediction
            predicted_class = load_model.predict(input_df)
            if predicted_class[0] == 0:
                st.success('Customer is more likely to churn')
            else:
                st.success('Customer is not likely to churn')

        except ValueError as e:
            st.error(f"Error in input data: {e}")