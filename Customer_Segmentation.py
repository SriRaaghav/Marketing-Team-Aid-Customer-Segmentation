import streamlit as st
import numpy as np
import pandas as pd
import joblib 

kmeans = joblib.load('/Users/raaghav/Desktop/ML-Projects/Customer_Segmentation/kmeans.pkl')
scaler = joblib.load('/Users/raaghav/Desktop/ML-Projects/Customer_Segmentation/scaler.pkl')

st.title("🧩 Customer Segmentation App")
st.write("Enter customer details to predict the segment.")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income ($)", min_value=0, max_value=200000, value=50000)
total_spending = st.number_input("Total Spending (sum of purchases)", min_value=0, max_value=50000, value=2000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=5)
num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0, max_value=100, value=15)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)


input_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'Total_Spendings': [total_spending],
        'NumWebPurchases': [num_web_purchases],
        'NumStorePurchases': [num_store_purchases],
        'NumWebVisitsMonth': [num_web_visits],
        'Recency': [recency]
    })


input_scaled = scaler.transform(input_data)

cluster_labels = {
    0: "Senior Value Seekers",
    1: "Affluent Traditionalists",
    2: "Premium Loyalists",
    3: "Omnichannel Enthusiasts",
    4: "Price-Sensitive Browsers",
    5: "Digital Window Shoppers"
}

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f'Predicted Cluster: {cluster_labels[cluster]}')

    
    



