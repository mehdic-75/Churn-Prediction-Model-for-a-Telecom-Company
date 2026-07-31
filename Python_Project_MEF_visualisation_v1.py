# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 20:57:39 2026

@author: mehdi
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import accuracy_score, classification_report, recall_score, roc_auc_score, roc_curve
import seaborn as sns
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


#Visualization
df.groupby("SeniorCitizen")["PhoneService"].value_counts().to_frame()
demographic_columns = ["gender", "SeniorCitizen", "Partner", "Dependents"]

colors = ['#8A7F8D', '#6B3F69']  

for col in demographic_columns:
    plt.figure(figsize=(8, 6))
    
    churn_rate = df.groupby(col)['Churn'].value_counts(normalize=True).unstack() * 100
    
    ax = churn_rate.plot(kind='bar', stacked=True, color=colors)
    
    plt.title(f'Churn Rate by {col}')
    plt.xlabel(col)
    plt.ylabel('Percentage (%)')
    

    if col == 'SeniorCitizen':
        plt.xticks([0, 1], ['Non-Senior', 'Senior'])
    
    plt.legend(['No Churn', 'Churned'])
    plt.tight_layout()
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt = ".2f")
plt.title("Correlation Heatmap")
plt.show()

