import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load and preprocess the dataset
data = pd.read_csv('UCI Cardiovascular.csv')

# Clean the dataset
data.replace(['?', '-9'], np.nan, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.mean())

# Convert target variable to binary
data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0)

# Split dataset into features and target
X = data.drop(columns=['num'])
y = data['num']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predict probabilities for the test set
xgb_y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate the XGBoost model using AUC
xgb_auc = roc_auc_score(y_test, xgb_y_pred_proba)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_y_pred_proba)

# Print AUC and other evaluation metrics
print(f"XGBoost AUC: {xgb_auc:.2f}")
print("\nXGBoost Confusion Matrix:")
print(confusion_matrix(y_test, xgb_model.predict(X_test)))
print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_model.predict(X_test)))

# Plot the ROC curve
plt.figure()
plt.plot(xgb_fpr, xgb_tpr, label=f"XGBoost (AUC = {xgb_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend(loc='lower right')
plt.show()
