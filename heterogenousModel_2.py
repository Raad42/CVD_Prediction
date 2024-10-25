import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'UCI Cardiovascular.csv'
data = pd.read_csv(file_path)

# Replace non-numeric values with NaN and then drop or fill them
data.replace('?', pd.NA, inplace=True)

# Convert the 'ca' column to numeric
data['ca'] = pd.to_numeric(data['ca'], errors='coerce')

# Drop rows with missing values
data_cleaned = data.dropna()

# Convert target variable 'num' into a binary classification (0 = no heart disease, 1 = heart disease)
data_cleaned.loc[:, 'heart_disease'] = data_cleaned['num'].apply(lambda x: 1 if x > 0 else 0)

# Drop the original 'num' and 'thal' columns
data_cleaned = data_cleaned.drop(columns=['num', 'thal'])

# Define the features (X) and the target (y)
X_cleaned = data_cleaned.drop(columns=['heart_disease'])
y_cleaned = data_cleaned['heart_disease']

# Split the cleaned data into training and testing sets (90% train, 10% test)
X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.1, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train_cleaned = scaler.fit_transform(X_train_cleaned)
X_test_cleaned = scaler.transform(X_test_cleaned)

# Define a function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate predicted probabilities for AUC and ROC curve
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    
    # Calculate accuracy, precision, recall, F1 score, and AUC
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Generate ROC curve values
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    return accuracy, precision, recall, f1, auc_score, conf_matrix, fpr, tpr

# Define the base learners without bagging
base_learners = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Artificial Neural Network (ANN)': MLPClassifier(random_state=42, max_iter=1000)
}

# Define parameter grids for each base learner for RandomizedSearchCV
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'K-Nearest Neighbors': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    'Artificial Neural Network (ANN)': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'],
        'alpha': [0.0001, 0.001, 0.01]
    }
}

# Create a list of tuples for base learners in stacking
base_models = [(name, model) for name, model in base_learners.items()]

# Define the meta-learner as MLP
meta_learner = MLPClassifier(random_state=42, max_iter=1000)

# Set up the Stacking Classifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_learner)

# Perform RandomizedSearchCV for base learners
optimized_base_models = {}
evaluation_results = []

for model_name, model in base_learners.items():
    print(f"Optimizing {model_name}...")
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grids[model_name],  # Define the parameter grid for the model
        n_iter=10,  # Adjust the number of iterations as needed
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train_cleaned, y_train_cleaned)
    optimized_base_models[model_name] = random_search.best_estimator_
    
    # Evaluate the optimized model
    accuracy, precision, recall, f1, auc_score, conf_matrix, fpr, tpr = evaluate_model(random_search.best_estimator_, X_train_cleaned, y_train_cleaned, X_test_cleaned, y_test_cleaned)
    
    # Append results for the base model
    evaluation_results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc_score,
        'Best Parameters': random_search.best_params_
    })
    
    print(f"{model_name} - Best Parameters: {random_search.best_params_}")

# Create a new stacking model with optimized base learners
optimized_base_models_list = [(name, model) for name, model in optimized_base_models.items()]
stacking_model_optimized = StackingClassifier(estimators=optimized_base_models_list, final_estimator=meta_learner)

# Fit the optimized stacking model
stacking_model_optimized.fit(X_train_cleaned, y_train_cleaned)

# Evaluate the optimized stacking model
accuracy, precision, recall, f1, auc_score, conf_matrix, fpr, tpr = evaluate_model(stacking_model_optimized, X_train_cleaned, y_train_cleaned, X_test_cleaned, y_test_cleaned)

# Append results for the meta-learner
evaluation_results.append({
    'Model': 'Stacking Model',
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'AUC': auc_score,
    'Best Parameters': "N/A (ensemble of optimized models)"
})

evaluation_df = pd.DataFrame(evaluation_results)

# Set pandas options to display all rows
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # No limit on width

# Print the evaluation summary table
print("\nEvaluation Summary Table:")
print(evaluation_df)

# Reset pandas options to default after displaying the DataFrame
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
pd.reset_option('display.width')

# Plot confusion matrix and ROC curve for the optimized stacking model
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Heart Disease', 'Heart Disease'], yticklabels=['No Heart Disease', 'Heart Disease'])
plt.title('Optimized Stacking Model - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f"Optimized Stacking Model (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line representing random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Optimized Stacking Model ROC Curve')
plt.legend(loc='lower right')
plt.show()