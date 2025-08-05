import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv('cardata.csv')

# drop car name
df.drop('Car_Name',axis=1,inplace = True)

# Apply One-Hot Encoding
df = pd.get_dummies(df,columns = ['Fuel_Type'	,'Seller_Type'	,'Transmission'],dtype = int)

# Separate features and target column

x = df.drop('Selling_Price',axis = 1)
y = df['Selling_Price']

#  Split data into 80% train and 20% test

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# ============================== Feature Scaling ==============================
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# ===================== Linear Regression Model =====================

# Train the Linear Regression Model
lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

# Evaluate Linear Regression model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mae)  # Note: Technically RMSE = sqrt(MSE), but using MAE here as per your code
r2 = r2_score(y_test, y_pred)



# Show Linear Regression results
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# ===================== Random Forest Regressor =====================

# Train Random Forest Model
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)
rf_pred = rf.predict(x_test)

# Evaluate Random Forest model
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mae)
rf_r2 = r2_score(y_test, rf_pred)

# # Show Random Forest results
# print(f"Random Forest MAE:{rf_mae:.2f}")
# print(f"Random Forest RMSE: {rf_rmse:.2f}")
# print(f"Random Forest R² Score: {rf_r2:.4f}")



# ===================== Feature Importance =====================

# Analyze important features from the Random Forest model
importances = rf.feature_importances_
feature_name = x.columns

feature_df = pd.DataFrame({'Feature': feature_name, "Importance": importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)



# ===================== Model Saving & Loading =====================

# Save the trained model to a file

joblib.dump(rf,"Hello.pkl")
joblib.dump(sc,"sc.pkl")


# Load the saved model
model = joblib.load('Hello.pkl')


print("✅ Model saved as phone_price_predictor.pkl")
# Predict again using the loaded model
y_pred_loaded = model.predict(x_test)

# Evaluate loaded model's performance
# rf_r2_y_pred_loaded = r2_score(y_test, y_pred_loaded)
# print(f"R² Score: {rf_r2_y_pred_loaded :.4f}")

print("=" * 100)


print(model.predict([[2017,	9.85,	6900,	0	,0,	0,	1	,1	,0	,0	,1]]))

print("=" * 100)

a = sc.fit_transform([[2017,	9.85,	6900,	0	,0,	0,	1	,1	,0	,0	,1]])

print(model.predict(a))