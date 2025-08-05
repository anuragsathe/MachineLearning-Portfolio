# ============================== Import Required Libraries ==============================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Sklearn libraries for ML models and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ============================== Load Dataset ==============================

df = pd.read_csv('clean_dataset.csv')

# Show full dataset
print(df)

# Check duplicated rows
print(df.duplicated().value_counts())

# Check unique values column-wise
print(df.nunique())

# Make a copy of original dataset
df1 = df.copy()

# Show data types of each column
df1.dtypes

# ============================== Data Overview ==============================

# Display head and tail of dataset
import re
print(df.head())
print(df.tail())

# Full info and stats of dataset
print(df.info())
print(df.nunique())
print(df.describe())

# ============================== Target Column Check ==============================

# Check value counts of target column 'Approved'
df['Approved'].value_counts()
print(df['Approved'].value_counts())

# Pie chart for distribution of classes
df['Approved'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(5, 5))
plt.show()

# ============================== Null Values Check ==============================

print(df.isnull().sum())  # Shows count of null values (use bfill, ffill, mean, median, mode if needed)

# Show data types again
print(df.dtypes)
print(df.head(2))

# ============================== Label Encoding for Categorical Columns ==============================

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Loop through columns and encode object (string) columns
for col in df.columns:
    if df[col].dtypes == object:
        df[col] = le.fit_transform(df[col])

# ============================== Feature-Label Split ==============================

x = df.drop('Approved', axis=1)  # Features
y = df['Approved']               # Target

# ============================== Train-Test Split ==============================

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# ============================== Feature Scaling ==============================

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test  = sc.fit_transform(x_test)

# ============================== Logistic Regression ==============================

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=42)
classifier.fit(x_train, y_train)

# Predict values using logistic regression
y_pred = classifier.predict(x_test)
y_pred

# Compare actual vs predicted
a = pd.DataFrame({'actual_value': y_test, 'predicted_value': y_pred})
a

# ============================== Evaluation - Logistic Regression ==============================

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(cm)

# Logistic Regression Accuracy
log_acc = accuracy_score(y_test, y_pred) * 100
log_acc

# Predict single data input using trained model
result = classifier.predict(np.array([[1, 22, 5.6, 1, 1, 5, 3, 4.5, 0, 2, 0, 1, 203, 450, 1]]))
result

# ============================== Random Forest Classifier ==============================

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_features=15)
rf.fit(x_train, y_train)

# Predict using Random Forest
y_pred_rf = rf.predict(x_test)
y_pred_rf

# Show actual vs predicted
pd.DataFrame({'actual_value': y_test, 'predicted_value': y_pred_rf})

# Evaluation - Random Forest
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm = confusion_matrix(y_test, y_pred_rf)
print(classification_report(y_test, y_pred_rf))
print(cm)

# Accuracy for Random Forest
rf_acc = accuracy_score(y_test, y_pred_rf) * 100
rf_acc

# Predict a sample using Random Forest model
result = rf.predict(np.array([[1, 22, 5.6, 1, 1, 5, 3, 4.5, 0, 2, 0, 1, 203, 450, 1]]))
result

# ============================== Support Vector Machine (SVM) ==============================

from sklearn import svm
svm = svm.SVC(kernel='linear', C=0.01)  # kernel can be: linear, rbf, poly, sigmoid
svm.fit(x_train, y_train)

# Predict using SVM
y_pred_svm = svm.predict(x_test)
y_pred_svm

# Compare actual vs predicted
pd.DataFrame({'actual_value': y_test, 'predicted_value': y_pred_svm})

# Evaluation - SVM
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm = confusion_matrix(y_test, y_pred_svm)
print(classification_report(y_test, y_pred_svm))
print(cm)

# Accuracy for SVM
svm_acc = accuracy_score(y_test, y_pred_svm) * 100
svm_acc

# Predict using SVM
result = svm.predict(np.array([[1, 29, 9, 0, 0, 6, 3, 4.5, 0, 3, 0, 1, 203, 590, 1]]))
result

# ============================== Accuracy Comparison Graph ==============================

plt.rcParams["figure.figsize"] = [8, 4]
plt.rcParams["figure.autolayout"] = True

# Labels and values
x = ['logistic accuracy', 'randomforest_accuracy', 'support vector machine_accuracy']
y = [log_acc, rf_acc, svm_acc]

width = 0.75
fig, ax = plt.subplots()

# Bar plot
pps = ax.bar(x, y, width, align='center')

# Show % on top of each bar
for p in pps:
    height = p.get_height()
    ax.text(x=p.get_x() + p.get_width() / 2, y=height + 1,
            s="{}%".format(height),
            ha='center')

plt.title('Accuracy of models')
plt.show()

# ============================== Save the Models ==============================

import joblib

# Save RandomForest model and StandardScaler
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(sc, 'scaler.pkl')
