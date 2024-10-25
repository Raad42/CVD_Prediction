import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, RocCurveDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
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

# Split the dataset into training (90%) and validation (10%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale the features using StandardScaler (normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define the models with specified parameters wrapped in BaggingClassifier
models_and_params = {
    'Bagging Logistic Regression': {
        'model': BaggingClassifier(estimator=LogisticRegression(C=0.01, random_state=42), n_estimators=10, random_state=42),
    },
    'Bagging Decision Tree': {
        'model': BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=3, min_samples_split=2, random_state=42), n_estimators=10, random_state=42),
    },
    'Bagging SVM': {
        'model': BaggingClassifier(estimator=SVC(C=1, kernel='linear', probability=True, random_state=42), n_estimators=10, random_state=42),
    },
    'Bagging K-Nearest Neighbors': {
        'model': BaggingClassifier(estimator=KNeighborsClassifier(n_neighbors=9, weights='distance'), n_estimators=10, random_state=42),
    },
    'Bagging Artificial Neural Network (ANN)': {
        'model': BaggingClassifier(estimator=MLPClassifier(hidden_layer_sizes=(100, 50), activation='tanh', solver='sgd', random_state=42, max_iter=1000), n_estimators=10, random_state=42),
    }
}

# Initialize results storage
results = []

# Perform K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model_info in models_and_params.items():
    model = model_info['model']
    
    # Perform K-Fold Cross Validation
    fold_accuracies = []
    for train_index, val_index in kf.split(X_train_scaled):
        X_fold_train, X_fold_val = X_train_scaled[train_index], X_train_scaled[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
        
        model.fit(X_fold_train, y_fold_train)
        y_fold_pred = model.predict(X_fold_val)
        accuracy = accuracy_score(y_fold_val, y_fold_pred)
        fold_accuracies.append(accuracy)

    # Average accuracy over all folds
    avg_accuracy = np.mean(fold_accuracies)
    
    # Train final model on the whole training data
    model.fit(X_train_scaled, y_train)

    # Evaluate on validation set (10% of UCI)
    y_val_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    val_accuracy = accuracy_score(y_val, y_val_pred)
    y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred_proba)
    f1 = f1_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)

    # Store results
    results.append({
        'Model': model_name,
        'Avg K-Fold Accuracy': avg_accuracy,
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

# Create a DataFrame for results and display it
results_df = pd.DataFrame(results)
print("Results Table:")
print(results_df)

# Create a summary table with relevant metrics for the Indian dataset
summary_results = pd.DataFrame({
    'Model': [result['Model'] for result in results],
    'Test Accuracy': [result['Test Accuracy'] for result in results],
    'Test AUC': [result['Test AUC'] for result in results],
    'F1 Score': [result['F1 Score'] for result in results],
    'Recall': [result['Recall'] for result in results],
    'Precision': [result['Precision'] for result in results],
})

print("Summary Results:")
print(summary_results)

# Create a bar chart comparing UCI and India dataset accuracy
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(results))

# Bar for UCI dataset accuracy
plt.bar(index, [result['Validation Accuracy'] for result in results], bar_width, label='UCI Dataset Accuracy', alpha=0.7)

# Bar for Indian dataset accuracy
plt.bar(index + bar_width, [result['Test Accuracy'] for result in results], bar_width, label='India Dataset Accuracy', alpha=0.7)

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison: UCI vs India Datasets')
plt.xticks(index + bar_width / 2, [result['Model'] for result in results], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
