import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
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
data_cleaned['heart_disease'] = data_cleaned['num'].apply(lambda x: 1 if x > 0 else 0)

# Drop the original 'num' column and the 'thal' column
data_cleaned = data_cleaned.drop(columns=['num', 'thal'])

# Define the features (X) and the target (y)
X_cleaned = data_cleaned.drop(columns=['heart_disease'])
y_cleaned = data_cleaned['heart_disease']

# Split the cleaned data into training and testing sets (80% train, 20% test)
X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

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
    y_prob = model.predict_proba(X_test)[:, 1]
    
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

# Model definitions with hyperparameters for RandomizedSearchCV
model_definitions = {
    'Logistic Regression': {
        'model': BaggingClassifier(estimator=LogisticRegression(random_state=42), n_estimators=10, random_state=42),
        'params': {
            'estimator__C': [0.01, 0.1, 1, 10, 100]
        }
    },
    'Decision Tree': {
        'model': BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=10, random_state=42),
        'params': {
            'estimator__max_depth': [3, 5, 10, None],
            'estimator__min_samples_split': [2, 5, 10]
        }
    },
    'SVM': {
        'model': BaggingClassifier(estimator=SVC(probability=True, random_state=42), n_estimators=10, random_state=42),
        'params': {
            'estimator__C': [0.1, 1, 10],
            'estimator__kernel': ['linear', 'rbf']
        }
    },
    'K-Nearest Neighbors': {
        'model': BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=10, random_state=42),
        'params': {
            'estimator__n_neighbors': [3, 5, 7, 9],
            'estimator__weights': ['uniform', 'distance']
        }
    },
    'Artificial Neural Network (ANN)': {
        'model': BaggingClassifier(estimator=MLPClassifier(random_state=42, max_iter=1000), n_estimators=10, random_state=42),
        'params': {
            'estimator__hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'estimator__activation': ['relu', 'tanh'],
            'estimator__solver': ['adam', 'sgd']
        }
    }
}

# Dictionary to store results
results = {}

# Evaluate all models with RandomizedSearchCV
for model_name, model_info in model_definitions.items():
    model = model_info['model']
    params = model_info['params']
    
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(model, params, n_iter=10, scoring='accuracy', random_state=42, n_jobs=-1)
    
    # Fit the model
    random_search.fit(X_train_cleaned, y_train_cleaned)
    
    # Print best parameters
    print(f"{model_name} Best Parameters: {random_search.best_params_}")
    
    # Evaluate the model
    accuracy, precision, recall, f1, auc_score, conf_matrix, fpr, tpr = evaluate_model(random_search, X_train_cleaned, y_train_cleaned, X_test_cleaned, y_test_cleaned)
    
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc_score,
        'conf_matrix': conf_matrix,
        'fpr': fpr,
        'tpr': tpr
    }
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc_score:.4f}")

# Plot confusion matrices
for model_name, result in results.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(result['conf_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=['No Heart Disease', 'Heart Disease'], yticklabels=['No Heart Disease', 'Heart Disease'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot combined ROC curve
plt.figure(figsize=(10, 8))
for model_name, result in results.items():
    plt.plot(result['fpr'], result['tpr'], label=f"{model_name} (AUC = {result['auc_score']:.2f})")

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line representing random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Collect the results in a list of dictionaries
model_results = []
for model_name, result in results.items():
    model_results.append({
        'Model': model_name,
        'Accuracy': f"{result['accuracy']:.4f}",
        'Precision': f"{result['precision']:.4f}",
        'Recall': f"{result['recall']:.4f}",
        'F1 Score': f"{result['f1_score']:.4f}",
        'AUC Score': f"{result['auc_score']:.4f}",
        'Best Parameters': str(random_search.best_params_)
    })

# Convert the list of dictionaries into a pandas DataFrame
results_df = pd.DataFrame(model_results)

# Print the table in a format that you can copy-paste
print(results_df.to_string(index=False))
