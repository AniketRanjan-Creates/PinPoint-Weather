import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from pycaret.regression import setup, create_model, tune_model, finalize_model, predict_model

# Step 1: Load and preprocess data
df = pd.read_csv("sorted_merged_weather_2010_2020.csv")
df = df.rename(columns={'prectotcorr': 'rainfall'})
df['date'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
df['month'] = df['MONTH']
df['dayofyear'] = df['date'].dt.dayofyear
df['season'] = df['month'] % 12 // 3

# Step 2: Feature Engineering
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

# Step 3: Split into train and test
df_train = df[df['YEAR'] < 2020].copy()
df_test = df[df['YEAR'] == 2020].copy()

features = [
    'LAT', 'LON', 'month', 'dayofyear', 'season',
    'qv2m', 'ts', 'ws10m',
    'rainfall_lag1', 'qv2m_lag1', 'ts_lag1', 'ws10m_lag1',
    'humidity_temp', 'wind_humidity', 'dayofyear_sin', 'dayofyear_cos'
]

# Step 4: User Input
user_lat = 22.6
user_lon = 85.2
user_input_date = '2020-12-03'
user_date = pd.to_datetime(user_input_date)

# Step 5: Nearest location match
locations = df_test[['LAT', 'LON']].drop_duplicates().to_numpy()
nearest = NearestNeighbors(n_neighbors=1).fit(locations)
_, idx = nearest.kneighbors([[user_lat, user_lon]])
matched_lat, matched_lon = locations[idx[0][0]]

# Step 6: Get closest date at that location
test_loc = df_test[(df_test['LAT'] == matched_lat) & (df_test['LON'] == matched_lon)].copy()
test_loc['date_diff'] = (test_loc['date'] - user_date).abs()
test_row = test_loc.loc[test_loc['date_diff'].idxmin()]

print(f"\nUsing LAT={test_row['LAT']}, LON={test_row['LON']}, Closest Date={test_row['date'].date()}")

# Step 7: PyCaret setup (memory optimized)
setup(
    data=df_train[features + ['rainfall_log']],
    target='rainfall_log',
    session_id=123,
    normalize=True,
    use_gpu=False,
    fold=3,
    n_jobs=1,
    verbose=False
)

# Step 8: Model creation and tuning
model = create_model('et')
model = tune_model(model, n_iter=2, fold=3, optimize='MAE')
final_model = finalize_model(model)

# Step 9: Prediction
X_input = test_row[features].to_frame().T
prediction_df = predict_model(final_model, data=X_input)
print("\n🔍 Prediction Output Columns:", prediction_df.columns.tolist())

if 'Label' in prediction_df.columns:
    pred_log = prediction_df['Label'].values[0]
elif 'prediction_label' in prediction_df.columns:
    pred_log = prediction_df['prediction_label'].values[0]
else:
    raise KeyError("❌ Prediction column not found in result.")

y_pred = np.expm1(pred_log)
y_true = test_row['rainfall']

# Step 10: Alert level based on deviation
error = abs(y_pred - y_true)
std_dev = df_train['rainfall'].std()

if error <= std_dev:
    alert = 'Low'
elif error <= 2 * std_dev:
    alert = 'Medium'
else:
    alert = 'High'

# Step 11: Output
print(f"\n✅ Prediction Summary:")
print(f"Predicted Rainfall: {y_pred:.2f} mm")
print(f"Actual Rainfall: {y_true:.2f} mm")
print(f"Alert Level: {alert}")
from pycaret.regression import save_model

save_model(final_model, 'best_rainfall_model')

