import pandas as pd

# Load the Indian cardiovascular dataset
india_file_path = 'Indian_Cardiovascular.csv'  # Replace with your actual file path
india_data = pd.read_csv(india_file_path)

# Display initial dataset information
print("\n--- Initial Summary of the Indian Dataset ---")
print(f"Total data points (rows): {india_data.shape[0]}")
print(f"Total features (columns): {india_data.shape[1]}")
print("\nMissing values per column before cleaning:\n", india_data.isnull().sum())

# Summary statistics before cleaning
print("\n--- Initial Dataset Statistics ---")
print(india_data.describe(include='all'))

# Replace '?' with NaN
india_data.replace('?', pd.NA, inplace=True)

# Drop rows with any missing values
india_data_cleaned = india_data.dropna()

# Summary of cleaned dataset
print("\n--- Summary After Removing Rows with Missing Values ---")
print(f"Total data points (rows) remaining: {india_data_cleaned.shape[0]}")
print(f"Total features (columns): {india_data_cleaned.shape[1]}")
print("\nMissing values per column in cleaned dataset:\n", india_data_cleaned.isnull().sum())

# Summary statistics after cleaning
print("\n--- Cleaned Dataset Statistics ---")
print(india_data_cleaned.describe(include='all'))
