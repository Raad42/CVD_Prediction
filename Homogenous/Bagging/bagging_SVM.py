import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
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

### Bagging Classifier with SVM ###
# Initialize the base estimator
svm_model = SVC(probability=True, random_state=42)

# Initialize and train the Bagging model
bagging_model = BaggingClassifier(estimator=svm_model, n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)

# Predict probabilities for the test set
bagging_y_pred_proba = bagging_model.predict_proba(X_test)[:, 1]

# Evaluate the Bagging model using AUC
bagging_auc = roc_auc_score(y_test, bagging_y_pred_proba)
bagging_fpr, bagging_tpr, _ = roc_curve(y_test, bagging_y_pred_proba)

# Print AUC and other evaluation metrics
print(f"Bagging Model with SVM AUC: {bagging_auc:.2f}")
print("\nBagging Model with SVM Confusion Matrix:")
print(confusion_matrix(y_test, bagging_model.predict(X_test)))
print("\nBagging Model with SVM Classification Report:")
print(classification_report(y_test, bagging_model.predict(X_test)))

# Plot the ROC curve for the Bagging model
plt.figure()
plt.plot(bagging_fpr, bagging_tpr, label=f"Bagging Model with SVM (AUC = {bagging_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Bagging Model with SVM')
plt.legend(loc='lower right')
plt.show()
