from pycaret.regression import *
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('sorted_merged_weather_2010_2020.csv')

# Basic EDA prints
print("Columns in dataset:", df.columns)
print("Unique YEAR values:", df['YEAR'].unique())
print("Unique MONTH values:", df['MONTH'].unique())
print("Unique DAY values:", df['DAY'].unique())
print("Missing YEAR:", df['YEAR'].isnull().sum())
print("Missing MONTH:", df['MONTH'].isnull().sum())
print("Missing DAY:", df['DAY'].isnull().sum())

# PyCaret setup for regression (predicting rainfall: 'prectotcorr')
reg = setup(
    data=df,
    target='prectotcorr',
    session_id=42,
    fold=5,
    verbose=False
)

# Compare and get the best model
best_model = compare_models()

# Save model comparison results to CSV
results_df = pull()
results_df.to_csv("model_comparison_results.csv", index=False)

# Plot the MAE of models from the results_df
plt.figure(figsize=(10,6))
plt.barh(results_df['Model'], results_df['MAE'], color='skyblue')
plt.xlabel('MAE (Mean Absolute Error)')
plt.title('Model Comparison: MAE of Different Models')
plt.gca().invert_yaxis()  # Highest MAE at the top
plt.show()

# Finalize and save the best model
final_model = finalize_model(best_model)
evaluate_model(final_model)
save_model(final_model, 'best_weather_model')

print("✅ Model training completed and saved.")
