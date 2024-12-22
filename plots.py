import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define data for each model type including test results
data = {
    'Model': ['ANN', 'SVM', 'KNN', 'LR', 'DT', 
              'ANN Bagging', 'SVM Bagging', 'KNN Bagging', 'LR Bagging', 'DT Bagging',
              'AdaBoost', 'Gradient Boosting', 'XGBoost', 
              'Stacking Model'],
    'Type': ['Standard'] * 5 + ['Bagging'] * 5 + ['Boosting'] * 3 + ['Stacking'],
    'AUC': [0.9120, 0.9051, 0.8924, 0.9074, 0.8808,
            0.9348, 0.9222, 0.9259, 0.9348, 0.9181,
            0.9502, 0.9197, 0.9423,
            0.9615],
    'Accuracy': [0.8333, 0.8000, 0.8333, 0.8833, 0.8333, 
                 0.8571, 0.8487, 0.8403, 0.8067, 0.8319,
                 0.8500, 0.8333, 0.8500,
                 0.8833],
    # Test data for each model (added)
    'India_AUC': [0.6499, 0.6845, 0.5352, 0.6499, 0.7003,
                 0.6521, 0.6112, 0.6868, 0.5417, 0.6811,
                 0.66, 0.60, 0.71,
                 0.6146],
    'India_Accuracy': [0.564, 0.550, 0.509, 0.564, 0.466,
                      0.563, 0.473, 0.548, 0.507, 0.566,
                      0.64, 0.60, 0.67,
                      0.555]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame to create a format suitable for plotting
df_melted_auc = df.melt(id_vars=['Model', 'Type'], value_vars=['AUC', 'India_AUC'],
                         var_name='AUC_Type', value_name='AUC_Score')
df_melted_accuracy = df.melt(id_vars=['Model', 'Type'], value_vars=['Accuracy', 'India_Accuracy'],
                              var_name='Accuracy_Type', value_name='Accuracy_Score')

# Set up the figure for the AUC bar chart with Accuracy line plot
plt.figure(figsize=(14, 8))
plt.title('Comparison of AUC and Accuracy Across Model Types')

# Plot AUC as a grouped bar chart
sns.barplot(data=df_melted_auc, x='Model', y='AUC_Score', hue='AUC_Type', dodge=True, palette='viridis')
plt.ylim(0.4, 1)  # Set y-axis limit to focus on the 0.4-1 range
plt.ylabel('Score (AUC and Accuracy)')
plt.xlabel('Models')

# Overlay the Accuracy line plot on the same y-axis
sns.lineplot(data=df_melted_accuracy, x='Model', y='Accuracy_Score', hue='Accuracy_Type',
             marker='o', color='darkblue')

# Add vertical lines to separate model types visually
model_type_indices = [4.5, 9.5, 12.5]  # Indices to add lines between model types
for idx in model_type_indices:
    plt.axvline(x=idx, color='grey', linestyle='--')

# Labeling and layout adjustments
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metric', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
