# -*- coding: utf-8 -*-
"""
Created on Sun Feb 8 22:06:36 2026

@author: mehdi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score
)

# Contexte

churn_csv = r"/Users/mehdi/Documents/Mehdi/HEC/Cours/Python/Project/Telco_Customer_Churn.csv"

df = pd.read_csv(churn_csv)

df['Churn'] = df['Churn'].map({'Yes':1,'No':0})

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

df = df.drop(columns=['customerID'])

binary_map = {'Yes':1,'No':0}

df['gender'] = df['gender'].map({'Female':1,'Male':0})
df['Partner'] = df['Partner'].map(binary_map)
df['Dependents'] = df['Dependents'].map(binary_map)
df['PhoneService'] = df['PhoneService'].map(binary_map)
df['PaperlessBilling'] = df['PaperlessBilling'].map(binary_map)

service_map = {'Yes':1,'No':0,'No internet service':0.5}
df['OnlineSecurity'] = df['OnlineSecurity'].map(service_map)
df['OnlineBackup'] = df['OnlineBackup'].map(service_map)
df['DeviceProtection'] = df['DeviceProtection'].map(service_map)
df['TechSupport'] = df['TechSupport'].map(service_map)
df['StreamingTV'] = df['StreamingTV'].map(service_map)
df['StreamingMovies'] = df['StreamingMovies'].map(service_map)

df['MultipleLines'] = df['MultipleLines'].map({'Yes':1,'No':0,'No phone service':0.5})
df['InternetService'] = df['InternetService'].map({'Fiber optic':1,'DSL':0.5,'No':0})
df['Contract'] = df['Contract'].map({'Month-to-month':0,'One year':1,'Two year':2})
df['PaymentMethod'] = df['PaymentMethod'].map({
    'Electronic check':0,
    'Mailed check':0.3,
    'Bank transfer (automatic)':0.7,
    'Credit card (automatic)':1
})

# ML

features = [
    'tenure','MonthlyCharges','TotalCharges',
    'Contract','OnlineSecurity','TechSupport',
    'InternetService','PaymentMethod',
    'PaperlessBilling','SeniorCitizen'
]

X = df[features]
y = df['Churn']



X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

background = shap.sample(X_train, 100)


model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3,
    eval_metric='logloss',
    use_label_encoder=False,
    base_score=0.5
)

model.fit(X_train, y_train)



Y_prob = model.predict_proba(X_test)[:,1]

threshold = 0.35
preds = (Y_prob > threshold).astype(int)



print("\nAccuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))

auc = roc_auc_score(y_test, Y_prob)
print("ROC AUC:", auc)


# Roc Curve

fpr,tpr,_ = roc_curve(y_test, Y_prob)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Churn Prediction")
plt.legend()
plt.show()


# Precision Recall Curve

p,r,_ = precision_recall_curve(y_test, Y_prob)

plt.figure()
plt.plot(r,p)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")
plt.show()


# Feature Importance

imp = pd.Series(model.feature_importances_, index=features).sort_values()

plt.figure(figsize=(8,6))
imp.head(10).plot(kind='barh')
plt.title("Top 10 Feature Importance")
plt.gca().invert_yaxis()
plt.show()

print("\nTop features:\n", imp.head(10))


# Shap Explainability
explainer = shap.TreeExplainer(model.get_booster())
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, plot_type="bar")

# Business List — Top Churn Risk Customers

risk = X_test.copy()
risk['Churn_Probability'] = Y_prob
risk = risk.sort_values('Churn_Probability', ascending=False)

print("\nTop 10 Customers at Risk:\n")
print(risk.head(10))
