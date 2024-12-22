import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define the data for the accuracy, AUC, F1 score, precision, and recall
data = {
    'Model': ['LR', 'DT', 'SVM', 'KNN', 'ANN', 
              'Stacking', 'Bagging LR', 'Bagging DT', 'Bagging SVM', 
              'Bagging KNN', 'Bagging ANN', 'AdaBoost', 'Gradient Boosting', 'XGBoost'],
    'Accuracy': [0.564, 0.466, 0.550, 0.509, 0.560,
                 0.555, 0.563, 0.473, 0.548, 0.507, 0.566, 0.640, 0.600, 0.670],
    'AUC': [0.6499, 0.7003, 0.6845, 0.5352, 0.6428,
            0.6146, 0.6521, 0.6112, 0.6868, 0.5417, 0.6811, 0.6600, 0.6000, 0.7100],
    'F1 Score': [0.8780, 0.8661, 0.8595, 0.8305, 0.8640,
                 0.8618, 0.8710, 0.8361, 0.8387, 0.8235, 0.8455, 0.7000, 0.6300, 0.6800],
    'Recall': [0.8852, 0.9016, 0.8524, 0.8032, 0.8852,
               0.8689, 0.8852, 0.8361, 0.8525, 0.8033, 0.8525, 0.6600, 0.7400, 0.8000],
    'Precision': [0.8709, 0.8333, 0.8666, 0.8596, 0.8437,
                  0.8548, 0.8571, 0.8361, 0.8254, 0.8448, 0.8387, 0.6900, 0.6800, 0.7300]
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Set up the figure
plt.figure(figsize=(14, 8))
plt.title('Comparison of Recall, Precision, and F1 Score Across Model Types (Indian Dataset)')

# Plot Recall as a bar chart
sns.barplot(data=df, x='Model', y='Recall', color='skyblue', label='Recall')
plt.ylabel('Score (Recall, Precision, F1 Score)')
plt.xlabel('Models')

# Overlay Precision and F1 Score as line plots on the same y-axis
sns.lineplot(data=df, x='Model', y='Precision', marker='o', color='green', label='Precision')
sns.lineplot(data=df, x='Model', y='F1 Score', marker='o', color='orange', label='F1 Score')

# Add vertical lines to separate model types visually (e.g., Standard, Bagging, Boosting, Stacking)
model_type_indices = [4.5, 5.5, 10.5, 11.5]  # Adjust indices as needed for visual clarity
for idx in model_type_indices:
    plt.axvline(x=idx, color='grey', linestyle='--')

# Labeling and layout adjustments
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metrics', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
