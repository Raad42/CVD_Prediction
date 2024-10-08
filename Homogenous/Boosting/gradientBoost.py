import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
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

# Initialize and train the Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Predict probabilities for the test set
gb_y_pred_proba = gb_model.predict_proba(X_test)[:, 1]

# Evaluate the Gradient Boosting model using AUC
gb_auc = roc_auc_score(y_test, gb_y_pred_proba)
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_y_pred_proba)

# Print AUC and other evaluation metrics
print(f"Gradient Boosting AUC: {gb_auc:.2f}")
print("\nGradient Boosting Confusion Matrix:")
print(confusion_matrix(y_test, gb_model.predict(X_test)))
print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, gb_model.predict(X_test)))

# Plot the ROC curve
plt.figure()
plt.plot(gb_fpr, gb_tpr, label=f"Gradient Boosting (AUC = {gb_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gradient Boosting')
plt.legend(loc='lower right')
plt.show()
