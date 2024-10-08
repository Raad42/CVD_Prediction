import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Cardiovascular_Disease_Dataset\Cardiovascular_Disease_Dataset.csv')

# Drop the 'patientid' column as it is not a useful feature for prediction
data = data.drop(columns=['patientid'])

# Check for missing values and fill them if necessary
data = data.fillna(data.mean())

# Split the dataset into features and target
X = data.drop(columns=['target'])
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Decision Tree ###
# Initialize and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict probabilities for the test set
dt_y_pred_proba = dt_model.predict_proba(X_test)[:, 1]

# Evaluate the Decision Tree model using AUC
dt_auc = roc_auc_score(y_test, dt_y_pred_proba)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_y_pred_proba)

# Print AUC and other evaluation metrics
print(f"Decision Tree AUC: {dt_auc:.2f}")
print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_model.predict(X_test)))
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_model.predict(X_test)))

# Plot the ROC curve for the Decision Tree model
plt.figure()
plt.plot(dt_fpr, dt_tpr, label=f"Decision Tree (AUC = {dt_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree')
plt.legend(loc='lower right')
plt.show()
