import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load and preprocess the dataset
data = pd.read_csv('UCI Cardiovascular.csv')

# Clean the dataset
data.replace(['?', '-9'], np.nan, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.mean())

# Convert target variable to binary
data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0)

# Split dataset into features and target
X = data.drop(columns=['num'])
y = data['num']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Define Base Models (Level-0) ###
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(probability=True, random_state=42)  # Set probability=True to output probabilities for AUC
dt = DecisionTreeClassifier(random_state=42)

### Define Meta-Model (Level-1) ###
meta_model = LogisticRegression()

# Create Stacking Classifier
stacking_model = StackingClassifier(
    estimators=[
        ('knn', knn),  # KNN base model
        ('svm', svm),  # SVM base model
        ('dt', dt)     # Decision Tree base model
    ],
    final_estimator=meta_model,  # Meta-model
    cv=5  # Use cross-validation to train base models
)

# Train the Stacking model
stacking_model.fit(X_train, y_train)

# Predict probabilities for the test set (needed for AUC)
y_pred_proba = stacking_model.predict_proba(X_test)[:, 1]

# Evaluate the AUC score
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Stacking Model AUC: {auc:.2f}")

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

# Additional Evaluation
y_pred = stacking_model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
