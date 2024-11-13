import streamlit as st
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine
import joblib
from store_manager_functions import *

# initialise data
engine = create_engine('sqlite:///store_data.db')

def load_data():
    query = "SELECT * FROM product_data"  
    return pd.read_sql(query, engine)
df = load_data()

# title/header
st.title("Store Manager Dashboard")
st.write("Account: Store Manager | Date: 2010 February 1st")

base_product = st.selectbox("Select Product", df['Base Product'].unique())
variation_details = df[(df['Base Product'] == base_product)]['Variation Detail'].unique()
variation_detail = st.selectbox("Select Variation Detail", variation_details)

st.write(f"**Selected Product:** {base_product} - {variation_detail}")

st.write("### Forecasted Demand")
lstm_results = lstm_prediction(df, base_product, variation_detail)
lstm_summary = lstm_results.dropna(subset=['Predicted Quantity'])
st.dataframe(lstm_summary[['Description','Date','Predicted Quantity']])

df, low_inventory_threshold, high_inventory_threshold = generate_inventory(lstm_results)
    
st.write("### Optimized Pricing")
pricing_results = dynamic_pricing(df, low_inventory_threshold, high_inventory_threshold)
st.dataframe(pricing_results[['Date', 'Predicted Quantity', 'Optimized Price', 'Optimized Predicted Revenue']])

st.write("### Demand Forecast (Units)")
st.line_chart(pricing_results.set_index('Date')[['Predicted Quantity']])

st.write("### Reveue Forecast (£)")
st.line_chart(pricing_results.set_index('Date')[['Optimized Predicted Revenue']])
    
st.write("### Price Adjustment (£ per Unit)")
st.line_chart(pricing_results.set_index('Date')[['Optimized Price']])
