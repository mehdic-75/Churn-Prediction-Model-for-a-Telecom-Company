# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 16:03:29 2026

@author: mehdi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve

import statsmodels.api as sm

# Data Loading

churn_csv = r"C:\Users\mehdi\Documents\Mehdi\HEC\Cours\Python\Project\Telco_Customer_Churn.csv"

df = pd.read_csv(churn_csv)

df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

df = df.drop(columns=['customerID'])

binary = {'Yes':1,'No':0}

df['gender'] = df['gender'].map({'Female':1,'Male':0})
df['Partner'] = df['Partner'].map(binary)
df['Dependents'] = df['Dependents'].map(binary)
df['PhoneService'] = df['PhoneService'].map(binary)
df['PaperlessBilling'] = df['PaperlessBilling'].map(binary)

service = {'Yes':1,'No':0,'No internet service':0.5}

df['OnlineSecurity'] = df['OnlineSecurity'].map(service)
df['OnlineBackup'] = df['OnlineBackup'].map(service)
df['DeviceProtection'] = df['DeviceProtection'].map(service)
df['TechSupport'] = df['TechSupport'].map(service)
df['StreamingTV'] = df['StreamingTV'].map(service)
df['StreamingMovies'] = df['StreamingMovies'].map(service)

df['MultipleLines'] = df['MultipleLines'].map({'Yes':1,'No':0,'No phone service':0.5})
df['InternetService'] = df['InternetService'].map({'Fiber optic':1,'DSL':0.5,'No':0})
df['Contract'] = df['Contract'].map({'Month-to-month':0,'One year':1,'Two year':2})

df['PaymentMethod'] = df['PaymentMethod'].map({
    'Electronic check':0,
    'Mailed check':0.3,
    'Bank transfer (automatic)':0.7,
    'Credit card (automatic)':1
})

# Features

features = [
    'tenure','MonthlyCharges','TotalCharges',
    'Contract','OnlineSecurity','TechSupport',
    'InternetService','PaymentMethod',
    'PaperlessBilling','SeniorCitizen'
]

X = df[features]
y = df['Churn']

# Standardization

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Logistic Regression

lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)

y_prob = lr.predict_proba(X_test)[:,1]
y_pred = lr.predict(X_test)

print("\nLOGISTIC REGRESSION PERFORMANCE\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", auc)

# ROC Curve

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression")
plt.legend()
plt.show()

# Economic Interpretation

X_sm = sm.add_constant(X_scaled)

logit = sm.Logit(y, X_sm)
result = logit.fit()

print(result.summary())

# Odds Ratios

odds = pd.DataFrame({
    "Feature": ["Intercept"] + features,
    "OddsRatio": np.exp(result.params)
}).sort_values("OddsRatio", ascending=False)

print("\nODDS RATIOS\n")
print(odds)

# Visualization Odds Ratios

plt.figure(figsize=(8,6))
sns.barplot(data=odds[1:], x="OddsRatio", y="Feature")
plt.title("Odds Ratios – Logistic Regression")
plt.axvline(1, linestyle="--")
plt.show()
