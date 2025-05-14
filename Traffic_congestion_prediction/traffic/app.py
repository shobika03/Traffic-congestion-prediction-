import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from helper import preprocess_data
from model_utils import train_random_forest, train_lstm

# ----------------- CACHE FUNCTIONS -----------------

@st.cache_data
def load_and_preprocess():
    return preprocess_data("C:\\MINIProject\\traffic\\Metro_Interstate_Traffic_Volume.csv")

@st.cache_resource
def load_rf_model(X, y):
    return train_random_forest(X, y)

@st.cache_resource
def load_lstm_model(X, y):
    return train_lstm(X, y)

@st.cache_data
def get_coordinates(location):
    try:
        geolocator = Nominatim(user_agent="traffic_predictor", timeout=10)
        return geolocator.geocode(f"{location}, Tamil Nadu, India")
    except GeocoderTimedOut:
        return None

# ----------------- LOAD DATA -----------------

X, y, df = load_and_preprocess()

# ----------------- SIDEBAR INPUT -----------------

st.sidebar.title("🚗 Traffic Congestion Predictor")
hour = st.sidebar.slider("Select Hour", 0, 23, 8)
day = st.sidebar.slider("Select Day of Month", 1, 31, 15)  # Added Day Selector
model_type = st.sidebar.radio("Select Model", ["Random Forest", "LSTM"])
city = st.sidebar.text_input("Enter City", value="Tiruchirappalli")
areas = st.sidebar.text_area("Enter Multiple Areas (one per line)", "Thillai Nagar\nCantonment\nSrirangam")
area_list = [area.strip() for area in areas.split('\n') if area.strip()]  # List of Areas

# ----------------- PREDICTION FUNCTION -----------------

def predict_traffic(input_row):
    input_data = [[
        input_row['hour'], input_row['day'], input_row['month'],
        input_row['dayofweek'], input_row['temp'],
        input_row['rain_1h'], input_row['snow_1h'], input_row['clouds_all']
    ]]
    
    if model_type == "Random Forest":
        model = load_rf_model(X, y)
        prediction = model.predict(input_data)[0]
    else:  # LSTM
        model, scaler_X, scaler_y = load_lstm_model(X, y)
        scaled_input = scaler_X.transform(input_data)
        scaled_input = scaled_input.reshape(1, 1, X.shape[1])
        prediction = scaler_y.inverse_transform(model.predict(scaled_input))[0][0]
    return prediction

# ----------------- GET GEOLOCATION -----------------

def get_location_data(area, city):
    full_location = f"{area}, {city}"
    location_data = get_coordinates(full_location)
    
    if location_data:
        return location_data.latitude, location_data.longitude
    else:
        return 13.0827, 80.2707  # Default to Chennai if location fails

# ----------------- DISPLAY RESULTS -----------------

st.title("🚦 Traffic Congestion Prediction")

# Hourly forecast for the day (0–23 hours)
st.markdown("### 📊 Hourly Forecast for Selected Day")
hourly_preds = []
for h in range(24):
    input_row = df[(df['hour'] == h) & (df['day'] == day)].iloc[0]
    prediction = predict_traffic(input_row)
    hourly_preds.append(prediction)

st.line_chart(hourly_preds)

# Congestion Level
input_row = df[(df['hour'] == hour) & (df['day'] == day)].iloc[0]
prediction = predict_traffic(input_row)
st.metric("Predicted Traffic Volume", f"{int(prediction)}")

if prediction > 4500:
    level = "🔴 High"
    color = "red"
elif prediction > 2500:
    level = "🟠 Medium"
    color = "orange"
else:
    level = "🟢 Low"
    color = "green"

st.markdown(f"### Congestion Level: **{level}**")

# ----------------- MAP: Multi-location Predictions -----------------

st.markdown("### 📍 Traffic Volume Predictions for Multiple Locations")
m = folium.Map(location=[13.0827, 80.2707], zoom_start=12)  # Default map center (Chennai)

for area in area_list:
    lat, lon = get_location_data(area, city)
    
    # Get traffic prediction for the current location
    input_row = df[(df['hour'] == hour) & (df['day'] == day)].iloc[0]
    prediction = predict_traffic(input_row)

    # Congestion level for the location
    if prediction > 4500:
        color = "red"
    elif prediction > 2500:
        color = "orange"
    else:
        color = "green"

    # Add marker for the area
    folium.Marker(
        [lat, lon],
        tooltip=f"{area}: {int(prediction)} vehicles",
        icon=folium.Icon(color=color)
    ).add_to(m)

st_folium(m, width=700)

# ----------------- WEATHER IMPACT (Optional) -----------------

st.markdown("### 🌦️ Weather Impact")
st.write("Weather conditions such as temperature, rain, and clouds can affect traffic patterns.")
weather_info = f"Temperature: {input_row['temp']}°C, Rain in last hour: {input_row['rain_1h']}mm, Cloudiness: {input_row['clouds_all']}%"
st.write(weather_info)

# ----------------- FEEDBACK BUTTON -----------------

st.markdown("### 📝 Feedback")
feedback = st.text_area("Please share your feedback about the predictions or app:", "")
if st.button("Submit Feedback"):
    if feedback:
        st.success("Thank you for your feedback!")
    else:
        st.warning("Please enter some feedback before submitting.")