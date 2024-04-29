import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="Employee Churn Prediction",
    layout='centered',
    page_icon=':1234:',
)

global churn_cutoff
churn_cutoff = 0.3

def app():
    # Loading the machine learning components
    ml_components_dict = tf.keras.models.load_model("./Assets/tel_model.h5")
    scaler_dict = joblib.load("./Assets/scalerKer.pkl")

    st.title(":1234: Employee Churn App")
    st.write("""Welcome to ChurnShield Employee Churn Prediction app!  
            This app allows you to predict the probability of Churn for a specific 
            employee based on our trained deep learning models.""")

    with st.form(key="information",clear_on_submit=True):
        st.write("Enter the information of your Employee")
        Age = st.number_input("Age of Employee in years? ")
        Gender = st.selectbox("Gender of Employee? ", ['Male', 'Female'])
        Education = st.selectbox("Education level of Employee? ", ['1. Below College', '2. College', '3. Bachelor', '4. Master', '5. Doctor'])
        MaritalStatus = st.selectbox("What's the Marital status of Employee? ", ['Single', 'Married', 'Divorced'])
        Department = st.selectbox("What's the Department of Employee? ", ['Sales', 'Research & Development', 'Human Resources'])
        JobRole = st.selectbox("What's the Job Role of Employee? ", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
        JobInvolvement = st.selectbox("Rate the Involvement of Employee? ", ['1. Low', '2. Medium', '3. High', '4. Very High'])
        JobStatisfaction = st.selectbox("Choose the level of Job statisfaction of Employee ?", ['1. Low', '2. Medium', '3. High', '4. Very High'])
        DistanceFromHome = st.number_input("What's the Distance from Home to Office of Employee in kms? ")
        OverTime = st.selectbox("Does Employee works Over Time? ", ['Yes', 'No'])
        BusinessTravel = st.selectbox("What's the Business Travel frequency of Employee? ", ['Frequently', 'Rarely', 'No Travel'])
        PercentSalaryHike = st.number_input("What's the Salary Hike of Employee in Percentage? ")
        YearsAtCompany = st.number_input("How many Years the Employee has worked in current Company? ")
        NumCompaniesWorked = st.number_input("Enter Number of Companies the Employee has worked previously? ")
        TotalWorkingYears = st.number_input("Enter Total Working years of Employee? ")
        WorkLifeBalance = st.selectbox("Rate the Work Life Balance of Employee: ", ['1. Low', '2. Medium', '3. High', '4. Very High'])
        RelationshipSatisfaction = st.selectbox("Rate the Relationship satifaction of Employee: ", ['1. Low', '2. Good', '3. Excellent', '4. Outstanding'])


        # Prediction
        if st.form_submit_button("Predict"):
            # Dataframe Creation
            data = pd.DataFrame({
                "DistanceFromHome": [DistanceFromHome],
                "Education": [Education],
                "JobInvolvement": [JobInvolvement],
                "JobStatisfaction": [JobStatisfaction],
                "NumCompaniesWorked": [NumCompaniesWorked],
                "OverTime": [OverTime],
                "PercentSalaryHike": [PercentSalaryHike],
                "RelationshipSatisfaction": [RelationshipSatisfaction],
                "TotalWorkingYears": [TotalWorkingYears],
                "WorkLifeBalance": [WorkLifeBalance],
                "YearsAtCompany": [YearsAtCompany],
                "Age": [Age],
                "BusinessTravel_Travel_Frequently": [BusinessTravel],
                "Department_Sales": [Department],
                "Gender_Male": [Gender],
                "JobRole_Sales Representative": [JobRole],
                "MaritalStatus_Single": [MaritalStatus]
            })       
        
        # Feature Engineering
        data['Gender_Male'] = 1 if data['Gender_Male'].iloc[0] == 'Male' else 0
        data['Education'] = 1 if data['Education'].iloc[0] == '1. Below College' else 2 if data['Education'].iloc[0] == '2. College' else 3 if data['Education'].iloc[0] == '3. Bachelor' else 4 if data['Education'].iloc[0] == '4. Master' else 5 if data['Education'].iloc[0] == '5. Doctor' else 1
        data['MaritalStatus_Single'] = 1 if data['MaritalStatus_Single'].iloc[0] == 'Single' else 0
        data['Department_Sales'] = 1 if data['Department_Sales'].iloc[0] == 'Sales' else 0
        data['JobRole_Sales Representative'] = 1 if data['JobRole_Sales Representative'].iloc[0] == 'Sales Representative' else 0
        data['JobInvolvement'] = 1 if data['JobInvolvement'].iloc[0] == '1. Low' else 2 if data['JobInvolvement'].iloc[0] == '2. Medium' else 3 if data['JobInvolvement'].iloc[0] == '3. High' else 4 if data['JobInvolvement'].iloc[0] == '4. Very High' else 1
        data['JobStatisfaction'] = 1 if data['JobStatisfaction'].iloc[0] == '1. Low' else 2 if data['JobStatisfaction'].iloc[0] == '2. Medium' else 3 if data['JobStatisfaction'].iloc[0] == '3. High' else 4 if data['JobStatisfaction'].iloc[0] == '4. Very High' else 1
        data['OverTime'] = 1 if data['OverTime'].iloc[0] == 'Yes' else 0
        data['BusinessTravel_Travel_Frequently'] = 1 if data['BusinessTravel_Travel_Frequently'].iloc[0] == 'Frequently' else 0
        data['WorkLifeBalance'] = 1 if data['WorkLifeBalance'].iloc[0] == '1. Low' else 2 if data['WorkLifeBalance'].iloc[0] == '2. Medium' else 3 if data['WorkLifeBalance'].iloc[0] == '3. High' else 4 if data['WorkLifeBalance'].iloc[0] == '4. Very High' else 1
        data['RelationshipSatisfaction'] = 1 if data['WorkLifeBalance'].iloc[0] == '1. Low' else 2 if data['RelationshipSatisfaction'].iloc[0] == '2. Good' else 3 if data['RelationshipSatisfaction'].iloc[0] == '3. Excellent' else 4 if data['RelationshipSatisfaction'].iloc[0] == '4. Outstanding' else 1

        # Scale the numerical columns
        data = scaler_dict.transform(data)
        st.write(data)

        # Make prediction using the model
        predictions = ml_components_dict.predict(data)
        # st.write(predictions)
        ans = predictions[0]
        # Display the predictions
        st.balloons()

        if ans>churn_cutoff:
            st.success(f"The Employee is likely to leave the Company.")
        else:
            st.success(f"The Employee is happy to be a part of Company.")
        
        # Display the predictions with custom styling
        # st.success(f"Predicted Churn Rate: {predictions[0]:,.2f}",icon="✅")