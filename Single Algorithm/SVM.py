import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'UCI Cardiovascular.csv'
data = pd.read_csv(file_path)

# Replace non-numeric values with NaN and then drop or fill them
data.replace('?', pd.NA, inplace=True)

# Convert the 'ca' and 'thal' columns to numeric
data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
data['thal'] = pd.to_numeric(data['thal'], errors='coerce')

# Drop rows with missing values
data_cleaned = data.dropna()

# Convert target variable 'num' into a binary classification (0 = no heart disease, 1 = heart disease)
data_cleaned['heart_disease'] = data_cleaned['num'].apply(lambda x: 1 if x > 0 else 0)

# Drop the original 'num' column
data_cleaned = data_cleaned.drop(columns=['num'])

# Define the features (X) and the target (y)
X_cleaned = data_cleaned.drop(columns=['heart_disease'])
y_cleaned = data_cleaned['heart_disease']

# Split the cleaned data into training and testing sets (80% train, 20% test)
X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train_cleaned = scaler.fit_transform(X_train_cleaned)
X_test_cleaned = scaler.transform(X_test_cleaned)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear', C=1)  # You can adjust the kernel and C value
svm_model.fit(X_train_cleaned, y_train_cleaned)

# Make predictions on the test set
y_pred_cleaned = svm_model.predict(X_test_cleaned)

# Evaluate the model
accuracy_cleaned = accuracy_score(y_test_cleaned, y_pred_cleaned)
classification_rep_cleaned = classification_report(y_test_cleaned, y_pred_cleaned)

# Generate and plot the confusion matrix
conf_matrix = confusion_matrix(y_test_cleaned, y_pred_cleaned)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Heart Disease', 'Heart Disease'], yticklabels=['No Heart Disease', 'Heart Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Output the results
print(f'Accuracy: {accuracy_cleaned}')
print(f'Classification Report:\n{classification_rep_cleaned}')
