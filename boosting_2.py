import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
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
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'param_distributions': {
            'n_estimators': randint(50, 200),
            'learning_rate': uniform(0.01, 1.0)
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'param_distributions': {
            'n_estimators': randint(50, 200),
            'learning_rate': uniform(0.01, 1.0),
            'max_depth': randint(1, 10),
            'min_samples_split': randint(2, 10)
        }
    },
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'param_distributions': {
            'n_estimators': randint(50, 200),
            'learning_rate': uniform(0.01, 1.0),
            'max_depth': randint(1, 10),
            'min_child_weight': randint(1, 10)
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

    # Prepare test data from India dataset
    X_test = india_df.drop(columns=['num'])  # Exclude target
    y_test = india_df['num']

    # Scale the test features using the same scaler
    X_test_scaled = scaler.transform(X_test)

    # Predict on the Indian test set
    y_test_pred = best_model.predict(X_test_scaled)

    # Calculate test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # Store results
    results.append({
        'Model': model_name,
        'Test Accuracy': test_accuracy,
        'Test AUC': test_auc,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1 Score': test_f1
    })

# Convert results to DataFrame for easier visualization
results_df = pd.DataFrame(results)

# Display results
print(results_df)

# Print out final metrics for each model
print("\nFinal Metrics for Each Model on Indian Test Dataset:")
for result in results:
    print(f"\nModel: {result['Model']}")
    print(f"Test Accuracy: {result['Test Accuracy']:.2f}")
    print(f"Test AUC: {result['Test AUC']:.2f}")
    print(f"Test Precision: {result['Test Precision']:.2f}")
    print(f"Test Recall: {result['Test Recall']:.2f}")
    print(f"Test F1 Score: {result['Test F1 Score']:.2f}")
