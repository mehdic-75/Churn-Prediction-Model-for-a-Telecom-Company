# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 20:57:39 2026

@author: mehdi
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import accuracy_score, classification_report, recall_score, roc_auc_score, roc_curve, precision_recall_curve
import seaborn as sns
import shap
import numpy as np
import matplotlib.pyplot as plt

# Contexte
churn_csv = r"C:\Users\mehdi\Documents\Mehdi\HEC\Cours\Python\Project\Telco_Customer_Churn.csv"

df = pd.read_csv(
    churn_csv,
    sep=',',
    engine='python',
    on_bad_lines='skip',
    encoding='utf-8'
)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

df = df.drop(columns = ['customerID'])

df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
df['MultipleLines'] = df['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service' : 0.5})
df['InternetService'] = df['InternetService'].map({'Fiber optic': 1, 'No': 0, 'DSL' : 0.5})
df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service' : 0.5})
df['OnlineBackup'] = df['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service' : 0.5})
df['DeviceProtection'] = df['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service' : 0.5})
df['TechSupport'] = df['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service' : 0.5})
df['StreamingTV'] = df['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service' : 0.5})
df['StreamingMovies'] = df['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service' : 0.5})
df['Contract'] = df['Contract'].map({'Month-to-month': 1, 'One year': 12, 'Two year' : 24})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0.5, 'Mailed check' : 0.25, 'Bank transfer (automatic)': 0.75,'Credit card (automatic)': 1})
print(df.describe())

print(df.head())

# ML

Y = df['Churn']
features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod']

X = df[features]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42, stratify = Y)

model = RandomForestClassifier(n_estimators = 300, random_state = 42, class_weight = 'balanced')
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
Y_prob = model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:", classification_report(Y_test, Y_pred))
auc = roc_auc_score(Y_test, Y_prob)
print("ROC AUC:", auc)

fpr, tpr, _ = roc_curve(Y_test, Y_prob)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Churn Prediction")
plt.legend()
plt.show()

# Precision Recall Curve

p,r,_ = precision_recall_curve(Y_test, Y_prob)

plt.figure()
plt.plot(r,p)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")
plt.show()

importances = pd.Series(model.feature_importances_, index=X.columns)\
                .sort_values(ascending=False)


# Shap Explainability

# use small sample for speed
X_shap = X_test.sample(300, random_state=42)

# Explainer pour RandomForestClassifier
explainer = shap.Explainer(model, X_train, model_output="probability")

shap_values = explainer(X_shap)

# Beeswarm plot
shap.plots.beeswarm(shap_values)

# Bar plot (global importance)
shap.plots.bar(shap_values)


plt.figure(figsize=(8,6))
importances.head(10).plot(kind='barh')
plt.title("Top 10 Feature Importance")
plt.gca().invert_yaxis()
plt.show()

print("\nTop features:\n", importances.head(10))


# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt = ".2f")
plt.title("Correlation Heatmap")
plt.show()

