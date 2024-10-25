import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
import matplotlib.pyplot as plt

# Load the two datasets
india_df = pd.read_csv('Indian_Cardiovascular.csv')  # Adjust the file name or path accordingly
uci_df = pd.read_csv('UCI Cardiovascular.csv')        # Adjust the file name or path accordingly

# Replace '?' with NaN to handle missing data
uci_df.replace('?', np.nan, inplace=True)
india_df.replace('?', np.nan, inplace=True)

# Convert numeric columns to the correct data types with error handling
def convert_to_numeric(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass  # Ignore columns that cannot be converted
    return df

uci_df = convert_to_numeric(uci_df)
india_df = convert_to_numeric(india_df)

# Rename columns if necessary (based on actual dataset)
uci_df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
india_df.columns = ['patientid', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'num']

# Drop unnecessary columns
india_df.drop(columns=['patientid'], inplace=True)

# Convert the 'num' column to binary (0 and 1)
uci_df['num'] = uci_df['num'].apply(lambda x: 1 if x > 0 else 0)
india_df['num'] = india_df['num'].apply(lambda x: 1 if x > 0 else 0)

# Handle missing values for numeric columns only for each dataset
numeric_columns_uci = uci_df.select_dtypes(include='number').columns
numeric_columns_india = india_df.select_dtypes(include='number').columns

uci_df[numeric_columns_uci] = uci_df[numeric_columns_uci].fillna(uci_df[numeric_columns_uci].mean())
india_df[numeric_columns_india] = india_df[numeric_columns_india].fillna(india_df[numeric_columns_india].mean())

# Convert categorical columns to numeric types if needed (if not already numeric)
uci_df['sex'] = uci_df['sex'].astype(int)  # Example for 'sex', apply to others if necessary
india_df['sex'] = india_df['sex'].astype(int)  # Example for 'sex', apply to others if necessary

# Data Preprocessing: Split the data into features (X) and target (y)
X = uci_df.drop(columns=['num', 'thal'])  # Exclude 'thal'
y = uci_df['num']

# Scale the features using StandardScaler (normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training (90%) and validation (10%) sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Define base learners
base_learners = [
    ('lr', LogisticRegression(solver='liblinear', C=0.1, random_state=42)),
    ('dt', DecisionTreeClassifier(min_samples_split=10, max_depth=10, random_state=42)),
    ('svm', SVC(kernel='rbf', C=1, probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(weights='uniform', n_neighbors=7)),
    ('ann', MLPClassifier(hidden_layer_sizes=(100,), alpha=0.0001, activation='tanh', max_iter=1000, random_state=42))
]

# Define the stacking ensemble model with ANN as the meta-learner
meta_learner = MLPClassifier(hidden_layer_sizes=(50,), activation='tanh', max_iter=1000, random_state=42)
stacked_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)

# Train the stacked model on the training data
stacked_model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = stacked_model.predict(X_val)
y_val_pred_proba = stacked_model.predict_proba(X_val)[:, 1]

# Calculate metrics for the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
auc = roc_auc_score(y_val, y_val_pred_proba)
f1 = f1_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)

# Prepare test data from India dataset
X_test = india_df.drop(columns=['num'])  # Exclude target
y_test = india_df['num']

# Scale the test features using the same scaler
X_test_scaled = scaler.transform(X_test)

# Predict on the Indian test set
y_test_pred = stacked_model.predict(X_test_scaled)
y_test_pred_proba = stacked_model.predict_proba(X_test_scaled)[:, 1]

# Calculate test metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_pred_proba)

# Create a results dictionary
results = {
    'Validation Accuracy': val_accuracy,
    'Validation AUC': auc,
    'Validation F1 Score': f1,
    'Validation Recall': recall,
    'Validation Precision': precision,
    'Test Accuracy': test_accuracy,
    'Test AUC': test_auc
}

# Create a DataFrame for results and display it
results_df = pd.DataFrame([results])
print("Results Table from Validation and Test Set:")
print(results_df)

# Create ROC AUC curves for UCI and India datasets
plt.figure(figsize=(12, 6))

# UCI ROC AUC
fpr_uci, tpr_uci, _ = roc_curve(y, stacked_model.predict_proba(X_scaled)[:, 1])
plt.plot(fpr_uci, tpr_uci, label='UCI ROC curve (area = {:.2f})'.format(auc), color='blue')

# India ROC AUC
fpr_india, tpr_india, _ = roc_curve(y_test, y_test_pred_proba)
plt.plot(fpr_india, tpr_india, label='India ROC curve (area = {:.2f})'.format(test_auc), color='orange')

# Plot the diagonal line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: UCI vs India Datasets')
plt.legend()
plt.grid()
plt.show()
