import pandas as pd
import streamlit as st
import datetime
import pickle

import seaborn as sns
import os
from io import BytesIO

try:
    from PIL import Image
except Exception:
    Image = None

car_df = pd.read_csv("cars24-car-price.csv")

# Header with optional logo displayed next to the title
logo_paths = ["logo.png", "assets/logo.png", os.path.join("car_pred", "logo.png"), "car_pred"]
logo_file = None
for p in logo_paths:
    if os.path.isfile(p):
        logo_file = p
        break

if logo_file:
    cols = st.columns([1, 7])
    with cols[0]:
        try:
            st.image(logo_file, width=80)
        except Exception:
            # If PIL is required for this file and isn't available, ignore image display
            st.write("")
    with cols[1]:
        st.markdown("# Cars24 Car Price Prediction and Analysis App")
        st.write(
            "This app predicts the selling price of a car based on its features and provides insights into the dataset."
        )
else:
    st.markdown("# Cars24 Car Price Prediction and Analysis App")
    st.write(
        "This app predicts the selling price of a car based on its features and provides insights into the dataset."
    )

#try:
    # Try using pandas Styler to hide the index in Streamlit
#    st.dataframe(car_df.head(10).style.hide_index())
#except Exception:
    # Fallback: reset index (keeps same visible numeric index in some Streamlit versions)
#    st.dataframe(car_df.head(10).reset_index(drop=True))

encode_dict = {
    "fuel_type": {"Diesel": 1, "Petrol": 2, "CNG": 3, "LPG": 4, "Electric": 5},
    "seller_type": {"Dealer": 1, "Individual": 2, "Trustmark Dealer": 3},
    "transmission_type": {"Manual": 1, "Automatic": 2},
}

def model_pred(fuel_type, transmission_type, engine, seats):
    # Loding the pre-trained model
    with open("car_pred", "rb") as file:
        reg_model = pickle.load(file)

    input_features = [[2018.0, 1, 40000, fuel_type, transmission_type, 19.70, engine, 86.30, seats]]
    return reg_model.predict(input_features)

# set the layout of the page

col1, col2 = st.columns(2)
fuel_type = col1.selectbox(
    " Select fuel type: ", ["Diesel", "Petrol", "CNG", "LPG", "Electric"]
)
engine = col1.slider("Set the engine power", 500, 5000, step=100)
transmission_type = col2.selectbox(
    " Select transmission type: ", ["Manual", "Automatic"]
)
seats = col2.selectbox("No of seats", [4, 5, 6])


if st.button("Predict Price"):
    fuel_type_encoded = encode_dict["fuel_type"][fuel_type]
    transmission_type_encoded = encode_dict["transmission_type"][transmission_type]

    price = model_pred(fuel_type_encoded, transmission_type_encoded, engine, seats)
    st.text("Predicted selling price in lakhs is: ₹ " + str(round(price[0], 2)))


#deployment on streamlit cloud