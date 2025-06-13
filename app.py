# app.py

import pandas as pd
from pycaret.regression import setup, compare_models, tune_model, finalize_model, save_model, predict_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Step 1: Load and Clean Data
df = pd.read_csv('sorted_merged_weather_2010_2020.csv')
df_sample = df.sample(frac=0.1, random_state=123).copy()

# Step 2: Feature Engineering
df['date'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
df['month'] = df['MONTH']
df['dayofyear'] = df['date'].dt.dayofyear
df['season'] = df['month'] % 12 // 3

def classify_alert(rainfall):
    if rainfall < 20:
        return 'light'
    elif rainfall < 100:
        return 'moderate'
    else:
        return 'heavy'

df['alert'] = df['prectotcorr'].apply(classify_alert)
df = df.drop(columns=['YEAR', 'MONTH', 'DAY', 'date'])

# Step 3: PyCaret Setup
regression_setup = setup(
    data=df,
    target='prectotcorr',
    session_id=123,
    normalize=True,
    categorical_features=['alert', 'season'],
    use_gpu=False
)

# Step 4: Compare and Choose Best Model
best_model = compare_models()
print("Best model found:", best_model)

# Step 5: Tune the Best Model with faster settings
tuned_model = tune_model(best_model, n_iter=3, fold=3, optimize='MAE', early_stopping=True)

# Step 6: Finalize and Save Model
final_model = finalize_model(tuned_model)
save_model(final_model, 'best_rainfall_model')

# Step 7: Evaluate Model
predictions = predict_model(final_model)
y_true = predictions['prectotcorr']
y_pred = predictions['Label']

print("R² Score:", r2_score(y_true, y_pred))
print("MAE:", mean_absolute_error(y_true, y_pred))
print("RMSE:", mean_squared_error(y_true, y_pred, squared=False))
