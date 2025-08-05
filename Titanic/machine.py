import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")
df.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket'], inplace=True)

# Clean Data
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df.dropna(subset=['Embarked'], inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)

# One-Hot Encoding
df = pd.get_dummies(df, columns=['Embarked', 'Sex'], dtype=int)

# Features & Target
x = df.drop('Survived', axis=1)
y = df['Survived']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)  # ← ✅ FIXED HERE

# -------------------------------
# Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print("Logistic Regression Report:\n", classification_report(y_test, y_pred))
cm_lr = confusion_matrix(y_test, y_pred)


# -------------------------------
# SVM (Default)
svm = SVC(kernel='rbf')
svm.fit(x_train, y_train)
y_pred_svm = svm.predict(x_test)

print("SVM Report:\n", classification_report(y_test, y_pred_svm))
cm_svm = confusion_matrix(y_test, y_pred_svm)


# -------------------------------
# Random Forest
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))

# -------------------------------
# Hyperparameter Tuning for SVM
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(x_train, y_train)

print("="*50)
print("Best Parameters:", grid.best_params_)

# Use Best Model
best_svm = grid.best_estimator_
y_pred_best_svm = best_svm.predict(x_test)

print("Tuned SVM Report:\n", classification_report(y_test, y_pred_best_svm))
print("Best SVM Accuracy after tuning:", accuracy_score(y_test, y_pred_best_svm) * 100)


# Save SVM model and StandardScaler

joblib.dump(best_svm,"SVM_model.pkl")
joblib.dump(sc,"Scaler.pkl")
