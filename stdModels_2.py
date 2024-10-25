import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier

# Load the two datasets
india_df = pd.read_csv('Indian_Cardiovascular.csv')  # Adjust the file name or path accordingly
uci_df = pd.read_csv('UCI Cardiovascular.csv')                # Adjust the file name or path accordingly

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
X_train = uci_df.drop(columns=['num', 'thal'])  # Exclude 'thal'
y_train = uci_df['num']
X_test = india_df.drop(columns=['num'])  # Exclude 'thal'
y_test = india_df['num']

# Scale the features using StandardScaler (normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define base models for stacking
base_models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100]
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    },
    'Artificial Neural Network (ANN)': {
        'model': MLPClassifier(random_state=42, max_iter=1000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd']
        }
    }
}

# Create a list of the base models and their parameters
estimators = []
for name, config in base_models.items():
    estimators.append((name, config['model']))

# Create a StackingClassifier
stacking_model = StackingClassifier(estimators=estimators, final_estimator=XGBClassifier(random_state=42))

# Set up RandomizedSearchCV for the stacking model
param_distributions = {
    'Logistic Regression__C': [0.01, 0.1, 1, 10, 100],
    'Decision Tree__max_depth': [3, 5, 10, None],
    'SVM__C': [0.1, 1, 10],
    'K-Nearest Neighbors__n_neighbors': [3, 5, 7, 9],
    'Artificial Neural Network (ANN)__hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'Artificial Neural Network (ANN)__activation': ['relu', 'tanh'],
    'Artificial Neural Network (ANN)__solver': ['adam', 'sgd']
}

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=stacking_model, param_distributions=param_distributions, n_iter=100, cv=5, scoring='roc_auc', n_jobs=-1)

# Train the model with RandomizedSearchCV
random_search.fit(X_train_scaled, y_train)

# Best parameters and score
print("Best Parameters: ", random_search.best_params_)
print("Best AUC Score during training: ", random_search.best_score_)

# Predict probabilities for the test set
stack_y_pred_proba = random_search.predict_proba(X_test_scaled)[:, 1]
stack_y_pred = random_search.predict(X_test_scaled)

# Evaluate the stacking model using AUC
stack_auc = roc_auc_score(y_test, stack_y_pred_proba)
stack_fpr, stack_tpr, _ = roc_curve(y_test, stack_y_pred_proba)

# Print AUC and other evaluation metrics
print(f"Stacking Model AUC: {stack_auc:.2f}")
print("\nStacking Model Confusion Matrix:")
print(confusion_matrix(y_test, stack_y_pred))
print("\nStacking Model Classification Report:")
print(classification_report(y_test, stack_y_pred))

# Plot the ROC curve
plt.figure()
plt.plot(stack_fpr, stack_tpr, label=f"Stacking Model (AUC = {stack_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Stacking Model with Randomized Search')
plt.legend(loc='lower right')
plt.show()
