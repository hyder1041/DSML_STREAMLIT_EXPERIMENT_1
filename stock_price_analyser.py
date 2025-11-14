import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.write(
    """
    # Stock Price Analyser

    Shown are the stock prices of Apple.
    """
)

ticker_symbol = st.text_input(
                        "Enter Stock Ticker Symbol", 
                        "AAPL",
                        help="For example: AAPL for Apple, MSFT for Microsoft",
                        key="placeholder")
# ticker_symbol = "AAPL"


col1, col2 = st.columns(2)

# Date input for selecting the date range
with col1:
    start_date = st.date_input("Select start date", 
                            value=datetime(2019, 1, 1), 
                            key="start_date")
with col2:
    end_date = st.date_input("Select end date", 
                            value=datetime(2024, 12, 31), 
                            key="end_date")

# Fetch the stock data from yfinance
ticker_data = yf.Ticker(ticker_symbol)
# Get the historical prices for the selected date range
ticker_df = ticker_data.history(start=f"{start_date}", 
                                end=f"{end_date}")

st.write(f"""
         ### Stock Data for {ticker_symbol} from {start_date} to {end_date}
         """)
# Display the stock data

st.dataframe(ticker_df.head())

# Plot the closing price chart

st.write("### Closing Price Chart")
st.line_chart(ticker_df['Close'])

# Plot the volume traded chart
st.write("### Volume of the Stock Traded")
st.line_chart(ticker_df['Volume'])