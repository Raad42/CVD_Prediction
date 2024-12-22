import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # For Artificial Neural Network (ANN)
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
file_path = 'UCI Cardiovascular.csv'
data = pd.read_csv(file_path)
# Replace non-numeric values with NaN and then drop or fill them
data.replace('?', pd.NA, inplace=True)
# Convert the 'ca' and 'thal' columns to numeric
data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
# Drop the 'thal' column
data_cleaned = data.drop(columns=['thal'])
# Drop rows with missing values
data_cleaned = data_cleaned.dropna()
# Convert target variable 'num' into a binary classification (0 = no heart disease, 1 = heart disease)
data_cleaned['heart_disease'] = data_cleaned['num'].apply(lambda x: 1 if x > 0 else 0)
# Drop the original 'num' column
data_cleaned = data_cleaned.drop(columns=['num'])
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
model_params = {
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
# Dictionary to store results
results = {}
# Perform Randomized Search and Evaluate Models
for model_name, mp in model_params.items():
    clf = RandomizedSearchCV(mp['model'], mp['params'], n_iter=10, cv=5, scoring='accuracy', random_state=42)
    accuracy, precision, recall, f1, auc_score, conf_matrix, fpr, tpr = evaluate_model(clf, X_train_cleaned, y_train_cleaned, X_test_cleaned, y_test_cleaned)
    
    results[model_name] = {
        'best_params': clf.best_params_,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc_score,
        'conf_matrix': conf_matrix,
        'fpr': fpr,
        'tpr': tpr
    }
    print(f"{model_name} - Best Params: {clf.best_params_}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC Score: {auc_score:.4f}")
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
        'Best Params': result['best_params'],
        'Accuracy': f"{result['accuracy']:.4f}",
        'Precision': f"{result['precision']:.4f}",
        'Recall': f"{result['recall']:.4f}",
        'F1 Score': f"{result['f1_score']:.4f}",
        'AUC Score': f"{result['auc_score']:.4f}"
    })
# Convert the list of dictionaries into a pandas DataFrame
results_df = pd.DataFrame(model_results)
# Print the table in a format that you can copy-paste
print(results_df.to_string(index=False))