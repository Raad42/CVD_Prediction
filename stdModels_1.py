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

# Define the models and their respective parameter grids for RandomizedSearchCV
models_and_params = {
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

# Initialize results storage
results = []

# Train and evaluate each model with RandomizedSearchCV
plt.figure(figsize=(10, 6))
for model_name, model_info in models_and_params.items():
    model = model_info['model']
    params = model_info['params']
    
    # Perform Randomized Search
    rand_search = RandomizedSearchCV(model, params, n_iter=50, cv=5, random_state=42, n_jobs=-1)
    rand_search.fit(X_train_scaled, y_train)

    # Get the best model
    best_model = rand_search.best_estimator_

    # Predict probabilities for the test set
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = best_model.predict(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # Store results
    results.append({
        'Model': model_name,
        'Best Parameters': rand_search.best_params_,
        'Accuracy': accuracy,
        'AUC': auc,
        'F1 Score': f1,
        'Recall': recall,
        'Precision': precision
    })

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")

# Plot settings
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of Different Models')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Adjust display options to show all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

# Create a DataFrame for results and display it
results_df = pd.DataFrame(results)
print(results_df)
