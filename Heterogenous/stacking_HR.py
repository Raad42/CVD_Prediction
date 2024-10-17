import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# Load and preprocess the dataset
data = pd.read_csv('UCI Cardiovascular.csv')

# Clean the dataset
data.replace(['?', '-9'], np.nan, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.mean())

# Convert target variable to binary
data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0)

# Sample a smaller portion of the dataset (e.g., 10%)
data_sampled = data.sample(frac=1, random_state=42)

# Split dataset into features and target
X = data_sampled.drop(columns=['num'])
y = data_sampled['num']

# Normalize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

### Define Base Models (Level-0) ###
knn = KNeighborsClassifier()
svm = SVC(probability=True, random_state=42)
dt = DecisionTreeClassifier(random_state=42)

### Define Meta-Model (Level-1) ###
meta_model = LogisticRegression()

# Create Stacking Classifier
stacking_model = StackingClassifier(
    estimators=[
        ('knn', knn),
        ('svm', svm),
        ('dt', dt)
    ],
    final_estimator=meta_model,
    cv=5
)

# Define parameter grid for each model
param_grid = {
    'knn__n_neighbors': [3, 5],
    'svm__C': [0.1, 1],
    'svm__kernel': ['linear', 'rbf'],
    'dt__max_depth': [None, 5],
    'final_estimator__C': [0.1, 1]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=stacking_model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Train the model with parameter tuning
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters: ", grid_search.best_params_)
print("Best AUC Score: ", grid_search.best_score_)

# Predict probabilities for the test set (needed for AUC)
y_pred_proba = grid_search.predict_proba(X_test)[:, 1]

# Evaluate the AUC score
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nStacking Model AUC: {auc:.2f}")

# Compute the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f"Stacking Model (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Stacking Model')
plt.legend(loc='lower right')
plt.show()

# Additional Evaluation: Accuracy, F1, Confusion Matrix
y_pred = grid_search.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")

### Calculate Model Diversity ###

# Predict with each base model individually
knn_pred = knn.fit(X_train, y_train).predict(X_test)
svm_pred = svm.fit(X_train, y_train).predict(X_test)
dt_pred = dt.fit(X_train, y_train).predict(X_test)

# Combine predictions into a matrix
pred_matrix = np.vstack((knn_pred, svm_pred, dt_pred)).T

# Compute pairwise diversity between classifiers
disagreements = pairwise_distances(pred_matrix, metric='hamming').mean()
print(f"\nModel Disagreement (Diversity): {disagreements:.2f}")
