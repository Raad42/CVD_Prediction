import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint

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

# Split the dataset into training (90%) and validation (10%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale the features using StandardScaler (normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define the models with parameter distributions for randomized search
models_and_params = {
    'Bagging Logistic Regression': {
        'model': BaggingClassifier(estimator=LogisticRegression(random_state=42), n_estimators=10, random_state=42),
        'param_distributions': {
            'estimator__C': uniform(0.001, 10),  # Regularization strength
            'estimator__solver': ['liblinear', 'saga'],  # Solvers to consider
        }
    },
    'Bagging Decision Tree': {
        'model': BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=10, random_state=42),
        'param_distributions': {
            'estimator__max_depth': randint(1, 10),  # Maximum depth of the tree
            'estimator__min_samples_split': randint(2, 10),  # Minimum samples to split an internal node
        }
    },
    'Bagging SVM': {
        'model': BaggingClassifier(estimator=SVC(probability=True, random_state=42), n_estimators=10, random_state=42),
        'param_distributions': {
            'estimator__C': uniform(0.1, 10),  # Regularization parameter
            'estimator__kernel': ['linear', 'rbf'],  # Kernel types to consider
        }
    },
    'Bagging K-Nearest Neighbors': {
        'model': BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=10, random_state=42),
        'param_distributions': {
            'estimator__n_neighbors': randint(1, 20),  # Number of neighbors
            'estimator__weights': ['uniform', 'distance'],  # Weight functions
        }
    },
    'Bagging Artificial Neural Network (ANN)': {
        'model': BaggingClassifier(estimator=MLPClassifier(random_state=42, max_iter=1000), n_estimators=10, random_state=42),
        'param_distributions': {
            'estimator__hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
            'estimator__activation': ['tanh', 'relu'],
        }
    }
}

# Initialize results storage
results = []

# Perform K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model_info in models_and_params.items():
    model = model_info['model']
    param_distributions = model_info['param_distributions']
    
    # Initialize RandomizedSearchCV
    randomized_search = RandomizedSearchCV(model, param_distributions, n_iter=10, scoring='accuracy', cv=kf, random_state=42, n_jobs=-1)
    
    # Fit the model with randomized search
    randomized_search.fit(X_train_scaled, y_train)

    # Get the best estimator from the randomized search
    best_model = randomized_search.best_estimator_
    
    # Train final model on the whole training data
    best_model.fit(X_train_scaled, y_train)

    # Evaluate on validation set (10% of UCI)
    y_val_pred = best_model.predict(X_val_scaled)
    
    # Calculate metrics
    val_accuracy = accuracy_score(y_val, y_val_pred)
    y_val_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred_proba)
    f1 = f1_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)

    # Store results
    results.append({
        'Model': model_name,
        'Best Params': randomized_search.best_params_,
        'Validation Accuracy': val_accuracy,
        'AUC': auc,
        'F1 Score': f1,
        'Recall': recall,
        'Precision': precision,
        'Val Predictions': y_val_pred_proba  # Store predictions for ROC Curve
    })

# Prepare test data from India dataset
X_test = india_df.drop(columns=['num'])  # Exclude target
y_test = india_df['num']

# Scale the test features using the same scaler
X_test_scaled = scaler.transform(X_test)

# Evaluate models on the Indian test dataset
for result in results:
    model_name = result['Model']
    model = models_and_params[model_name]['model']
    
    # Train the best model again to ensure it's fitted
    best_params = result['Best Params']
    model.set_params(**best_params)
    model.fit(X_train_scaled, y_train)
    
    # Predict on the Indian test set
    y_test_pred = model.predict(X_test_scaled)
    y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    # Update the results with test metrics
    result['Test Accuracy'] = test_accuracy
    result['Test AUC'] = test_auc
    result['Test Predictions'] = y_test_pred_proba  # Store predictions for ROC Curve

# Convert results to DataFrame for easier visualization
results_df = pd.DataFrame(results)

# Display results
print(results_df)

# Plot ROC Curves for each model
plt.figure(figsize=(12, 8))
for result in results:
    fpr, tpr, _ = roc_curve(y_test, result['Test Predictions'])
    plt.plot(fpr, tpr, label=f"{result['Model']} (AUC = {result['Test AUC']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Different Models')
plt.legend(loc='lower right')
plt.show()
