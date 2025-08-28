import streamlit as st
import pandas as pd
import joblib
model=joblib.load("D:\Fraud-Detection\model\\rlf_pipeline.pkl")
st.title("Fraud Detection")
st.markdown("Please enter the Transaction Details and use the predict button")
st.divider() #for better look
transcation_type=st.selectbox("Transaction Type",['PAYMENT','TRANSFER','CASH_OUT','DEPOSIT'])
amount=st.number_input("Amount",min_value=0.0,value=1000.0)
oldbalanceOrg=st.number_input("Old Balance (Sender)",min_value=0.0,value=10000.0)
newbalanceOrig=st.number_input("New Balance (Sender)",min_value=0.0,value=9000.0)
oldbalanceDest=st.number_input("Old Balance (Receiver)",min_value=0.0,value=0.0)
newbalanceDest=st.number_input("New Balance (Receiver)",min_value=0.0,value=0.0)
if st.button("Predict"):
    input_data=pd.DataFrame([{
        "type":transcation_type,
        "amount":amount,
        "oldbalanceOrg":oldbalanceOrg,
        "newbalanceOrig":newbalanceOrig,
        "oldbalanceDest":oldbalanceDest,
        "newbalanceDest":newbalanceDest
    }])
    input_data['balanceDiffOrig']=input_data['oldbalanceOrg']-input_data['newbalanceOrig']
    input_data['balanceDiffDest']=input_data['newbalanceDest']-input_data['oldbalanceDest']
    prediction=model.predict(input_data)[0] #taking the first element or index

    st.subheader(f"Prediction :{int(prediction)}")

    if prediction==1:
        st.error("this transaction is fraudulent")
    else:
        st.success("this transaction is legitimate")