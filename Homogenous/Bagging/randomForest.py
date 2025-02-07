import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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

# Normalize the feature data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

### Random Forest Model with GridSearchCV ###
# Define the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Set up GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Train the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters from GridSearchCV
print("Best Parameters from GridSearchCV: ", grid_search.best_params_)

# Use the best model from grid search
best_rf_model = grid_search.best_estimator_

# Predict probabilities for the test set using the best model
rf_y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]

# Evaluate the Random Forest model using AUC
rf_auc = roc_auc_score(y_test, rf_y_pred_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_pred_proba)

# Print AUC and other evaluation metrics
print(f"Random Forest AUC: {rf_auc:.2f}")
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, best_rf_model.predict(X_test)))
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, best_rf_model.predict(X_test)))

# Plot the ROC curve
plt.figure()
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend(loc='lower right')
plt.show()
