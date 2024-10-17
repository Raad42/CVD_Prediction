import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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

# Define the models to be tested for bagging
models = {
    'SVM': SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Set up hyperparameter grids for each model
param_grids = {
    'SVM': {'estimator__C': [0.1, 1, 10], 'estimator__kernel': ['linear', 'rbf']},
    'Logistic Regression': {'estimator__C': [0.1, 1, 10]},
    'KNN': {'estimator__n_neighbors': [3, 5, 7]},
    'Decision Tree': {'estimator__max_depth': [3, 5, 10]}
}

# Function to run GridSearch for a specific model
def grid_search_bagging(model_name, model, param_grid):
    bagging = BaggingClassifier(estimator=model, n_estimators=100, random_state=42)
    grid_search = GridSearchCV(bagging, param_grid=param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n{model_name} Bagging Model AUC: {auc:.2f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_model.predict(X_test)))
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))
    return auc, best_model

# Test all models for bagging and print results
best_auc = 0
best_model_name = None
best_model = None

for model_name, model in models.items():
    print(f"\n--- Testing {model_name} for Bagging ---")
    auc, trained_model = grid_search_bagging(model_name, model, param_grids[model_name])
    if auc > best_auc:
        best_auc = auc
        best_model_name = model_name
        best_model = trained_model

print(f"\nBest Bagging Model: {best_model_name} with AUC = {best_auc:.2f}")

# Plot ROC curve for the best model
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)

plt.figure()
plt.plot(fpr, tpr, label=f"{best_model_name} (AUC = {best_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for Best Bagging Model: {best_model_name}')
plt.legend(loc='lower right')
plt.show()

### Random Forest Model (as in the original code) ###

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict probabilities for the test set
rf_y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluate the Random Forest model using AUC
rf_auc = roc_auc_score(y_test, rf_y_pred_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_pred_proba)

# Print AUC and other evaluation metrics
print(f"\nRandom Forest AUC: {rf_auc:.2f}")
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_model.predict(X_test)))
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_model.predict(X_test)))

# Plot the ROC curve
plt.figure()
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend(loc='lower right')
plt.show()
