import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from pycaret.regression import load_model, predict_model

# Load trained model (make sure this file exists)
model = load_model("best_rainfall_model")

# Load dataset
@st.cache_data

def load_data():
    df = pd.read_csv("sorted_merged_weather_2010_2020.csv")
    df = df.rename(columns={'prectotcorr': 'rainfall'})
    df['date'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    df['month'] = df['MONTH']
    df['dayofyear'] = df['date'].dt.dayofyear
    df['season'] = df['month'] % 12 // 3
    df = df.sort_values(by=['LAT', 'LON', 'date'])
    df['rainfall_lag1'] = df.groupby(['LAT', 'LON'])['rainfall'].shift(1)
    df['qv2m_lag1'] = df.groupby(['LAT', 'LON'])['qv2m'].shift(1)
    df['ts_lag1'] = df.groupby(['LAT', 'LON'])['ts'].shift(1)
    df['ws10m_lag1'] = df.groupby(['LAT', 'LON'])['ws10m'].shift(1)
    df['humidity_temp'] = df['qv2m'] * df['ts']
    df['wind_humidity'] = df['qv2m'] * df['ws10m']
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    df['rainfall_log'] = np.log1p(df['rainfall'])
    df = df.dropna()
    return df

df = load_data()
df_test = df[df['YEAR'] == 2020].copy()
df_train = df[df['YEAR'] < 2020].copy()

# UI
st.title("🌧️ Rainfall Prediction (Based on 2010–2019 Data)")

user_lat = st.number_input("Enter Latitude", value=22.6)
user_lon = st.number_input("Enter Longitude", value=85.2)
user_date = st.date_input("Enter Date (in 2020)", value=datetime(2020, 6, 15))

if st.button("Predict"):
    locations = df_test[['LAT', 'LON']].drop_duplicates().to_numpy()
    nearest = NearestNeighbors(n_neighbors=1).fit(locations)
    _, idx = nearest.kneighbors([[user_lat, user_lon]])
    matched_lat, matched_lon = locations[idx[0][0]]

    subset = df_test[(df_test['LAT'] == matched_lat) & (df_test['LON'] == matched_lon)].copy()
    subset['date_diff'] = (subset['date'] - pd.to_datetime(user_date)).abs()
    if subset.empty:
        st.error("No data available for this location/date")
    else:
        row = subset.loc[subset['date_diff'].idxmin()]
        features = [
            'LAT', 'LON', 'month', 'dayofyear', 'season',
            'qv2m', 'ts', 'ws10m',
            'rainfall_lag1', 'qv2m_lag1', 'ts_lag1', 'ws10m_lag1',
            'humidity_temp', 'wind_humidity', 'dayofyear_sin', 'dayofyear_cos'
        ]
        X_input = row[features].to_frame().T

        pred_df = predict_model(model, data=X_input)
        pred_log = pred_df['Label'].values[0] if 'Label' in pred_df.columns else pred_df['prediction_label'].values[0]
        predicted = np.expm1(pred_log)
        actual = row['rainfall']

        # Alert level
        std_dev = df_train['rainfall'].std()
        error = abs(predicted - actual)
        if error <= std_dev:
            alert = 'Low'
        elif error <= 2 * std_dev:
            alert = 'Medium'
        else:
            alert = 'High'

        st.success(f"✅ Prediction Summary for {row['date'].date()} at LAT={matched_lat}, LON={matched_lon}")
        st.markdown(f"**Predicted Rainfall:** {predicted:.2f} mm")
        st.markdown(f"**Actual Rainfall:** {actual:.2f} mm")
        st.markdown(f"**Alert Level:** {alert}")
