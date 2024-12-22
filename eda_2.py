# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load the datasets (adjust paths to your files)
india_df = pd.read_csv('Indian_Cardiovascular.csv')  # Change this to the correct path
uci_df = pd.read_csv('UCI Cardiovascular.csv')       # Change this to the correct path

# Replace '?' with NaN and convert relevant columns to numeric
uci_df.replace('?', np.nan, inplace=True)
india_df.replace('?', np.nan, inplace=True)

# Convert columns to numeric types where applicable
def convert_to_numeric(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass
    return df

uci_df = convert_to_numeric(uci_df)
india_df = convert_to_numeric(india_df)

# Rename columns to ensure consistency
uci_df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
india_df.columns = ['patientid', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'num']

# Drop unnecessary columns (patientid from India dataset)
india_df.drop(columns=['patientid'], inplace=True)
uci_df.drop(columns=['thal'], inplace=True)

# Convert the 'num' column to binary for both datasets (presence or absence of CVD)
uci_df['num'] = uci_df['num'].apply(lambda x: 1 if x > 0 else 0)
india_df['num'] = india_df['num'].apply(lambda x: 1 if x > 0 else 0)

# Handle missing values: fill numeric columns with column means
uci_df.fillna(uci_df.mean(), inplace=True)
india_df.fillna(india_df.mean(), inplace=True)

# Identify which features are ordinal
ordinal_features = ['sex', 'fbs', 'slope', 'ca']

# Ensure ordinal features are treated as categorical (we'll keep them as integers)
uci_df[ordinal_features] = uci_df[ordinal_features].astype('category')
india_df[ordinal_features] = india_df[ordinal_features].astype('category')

# Continuous features to be scaled
continuous_features = ['age', 'cp', 'trestbps', 'chol', 'restecg', 'thalach', 'exang', 'oldpeak']

# Compare feature distributions between UCI and India datasets using KDE plots for continuous features
for feature in continuous_features:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(uci_df[feature], label='UCI Dataset', shade=True)
    sns.kdeplot(india_df[feature], label='India Dataset', shade=True)
    plt.title(f'Distribution of {feature} in UCI vs India Dataset')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Statistical testing: Kolmogorov-Smirnov Test for continuous feature distribution differences
print("Kolmogorov-Smirnov Test Results (Continuous Features):")
for feature in continuous_features:
    stat, p_value = ks_2samp(uci_df[feature].dropna(), india_df[feature].dropna())
    print(f"{feature}: KS-statistic = {stat:.4f}, p-value = {p_value:.4f}")

# Set display options to show the full correlation matrix without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Full Correlation matrix for UCI dataset
print("\nFull Correlation Matrix for UCI Dataset:")
uci_corr_full = uci_df.corr()
print(uci_corr_full)

# Plot heatmap for UCI correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(uci_corr_full, annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Full Correlation Heatmap - UCI Dataset")
plt.show()

# Full Correlation matrix for India dataset
print("\nFull Correlation Matrix for India Dataset:")
india_corr_full = india_df.corr()
print(india_corr_full)

# Plot heatmap for India correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(india_corr_full, annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Full Correlation Heatmap - India Dataset")
plt.show()

# Reset display options if needed
pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')

# Split the UCI and India datasets into features (X) and target (y), keeping ordinal features intact
X_uci = uci_df.drop(columns=['num'])
y_uci = uci_df['num']

X_india = india_df.drop(columns=['num'])
y_india = india_df['num']

# Split UCI data into training and validation sets (90% train, 10% validation)
X_train_uci, X_val_uci, y_train_uci, y_val_uci = train_test_split(X_uci, y_uci, test_size=0.1, random_state=42)

# Standardize (scale) the continuous features only
scaler = StandardScaler()
X_train_uci_scaled = X_train_uci.copy()
X_val_uci_scaled = X_val_uci.copy()
X_india_scaled = X_india.copy()

X_train_uci_scaled[continuous_features] = scaler.fit_transform(X_train_uci[continuous_features])
X_val_uci_scaled[continuous_features] = scaler.transform(X_val_uci[continuous_features])
X_india_scaled[continuous_features] = scaler.transform(X_india[continuous_features])

# Train Logistic Regression on UCI dataset (ordinal features are treated as categories)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_uci_scaled, y_train_uci)

# Evaluate Logistic Regression on UCI dataset (in-sample performance)
y_pred_uci = lr_model.predict(X_val_uci_scaled)
print("UCI (in-sample) Logistic Regression Accuracy:", accuracy_score(y_val_uci, y_pred_uci))
print("UCI (in-sample) Logistic Regression AUC:", roc_auc_score(y_val_uci, y_pred_uci))

# Evaluate Logistic Regression on Indian dataset (out-of-sample performance)
y_pred_india = lr_model.predict(X_india_scaled)
print("India (out-of-sample) Logistic Regression Accuracy:", accuracy_score(y_india, y_pred_india))
print("India (out-of-sample) Logistic Regression AUC:", roc_auc_score(y_india, y_pred_india))

## Train RandomForest on UCI dataset (ordinal features naturally handled by tree-based models)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_uci_scaled, y_train_uci)

# Evaluate RandomForest on UCI dataset
y_pred_uci_rf = rf_model.predict(X_val_uci_scaled)
print("UCI (in-sample) RandomForest Accuracy:", accuracy_score(y_val_uci, y_pred_uci_rf))
print("UCI (in-sample) RandomForest AUC:", roc_auc_score(y_val_uci, y_pred_uci_rf))

# Evaluate RandomForest on Indian dataset (out-of-sample performance)
y_pred_india_rf = rf_model.predict(X_india_scaled)
print("India (out-of-sample) RandomForest Accuracy:", accuracy_score(y_india, y_pred_india_rf))
print("India (out-of-sample) RandomForest AUC:", roc_auc_score(y_india, y_pred_india_rf))

# Feature importance from RandomForest for India dataset
importances_india = rf_model.feature_importances_
indices_india = np.argsort(importances_india)[::-1]

print("\nFeature Importances (India Dataset):")
for i in indices_india:
    print(f"{X_india.columns[i]}: {importances_india[i]:.4f}")

plt.figure(figsize=(10, 6))
plt.title("RandomForest Feature Importances (India Dataset)")
plt.bar(range(X_india.shape[1]), importances_india[indices_india], align="center")
plt.xticks(range(X_india.shape[1]), [X_india.columns[i] for i in indices_india], rotation=90)
plt.ylabel('Importance')
plt.xlabel('Features')
plt.show()
