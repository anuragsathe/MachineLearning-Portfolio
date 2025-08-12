# ===================== Import Required Libraries =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

# ===================== Data Loading & Preprocessing =====================
df = pd.read_csv("machine_learning_dataset.csv")

# Drop unnecessary columns
df.drop(['Phone Name', 'Color', 'Display Type', 'ROM (GB)',"Brand"], axis=1, inplace=True)

# Apply Label Encoding on categorical columns
le = LabelEncoder()
# df['Brand'] = le.fit_transform(df['Brand'])
df['Back Camera'] = le.fit_transform(df['Back Camera'])
df['Front Camera'] = le.fit_transform(df['Front Camera'])
df['Processor'] = le.fit_transform(df['Processor'])

# ===================== Train-Test Split =====================
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== Random Forest Regressor =====================
print("=" * 100)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mae)  # Note: Usually RMSE = sqrt(MSE), but using sqrt(MAE) as in your code
rf_r2 = r2_score(y_test, rf_pred)

print(f"Random Forest MAE: ₹{rf_mae:.2f}")
print(f"Random Forest RMSE: ₹{rf_rmse:.2f}")
print(f"Random Forest R² Score: {rf_r2:.4f}")

# ===================== Feature Importance =====================
print("=" * 100)

importances = rf.feature_importances_
feature_names = X.columns

feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)
print(feature_df)

# Plot Top 10 Features
feat_imp = pd.Series(importances, index=feature_names)
feat_imp.nlargest(10).plot(kind='barh', figsize=(10, 6))
plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Feature Importance Score")
plt.tight_layout()
plt.show()

# ===================== Model Saving & Loading =====================
print("=" * 100)

joblib.dump(rf, 'phone_price_predictor.pkl')
print("✅ Model saved as phone_price_predictor.pkl")

model = joblib.load('phone_price_predictor.pkl')
y_pred_loaded = model.predict(X_test)

rf_r2_loaded = r2_score(y_test, y_pred_loaded)
print(f"R² Score (Loaded Model): {rf_r2_loaded:.4f}")

print("=" * 100)
