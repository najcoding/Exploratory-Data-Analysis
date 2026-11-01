import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load the Dataset
# Replace 'your_dataset.csv' with the actual path to your file
df = pd.read_csv("your_dataset.csv")

# 2. Inspect the Structure
print("--- First 5 Rows ---")
print(df.head())

print("\n--- Data Shapes and Types ---")
print(df.info())

print("\n--- Summary Statistics (Numerical columns) ---")
print(df.describe())

# 3. Data Cleaning: Check for Missing Values & Duplicates
print("\n--- Missing Values Per Column ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows Count ---")
print(df.duplicated().sum())

# 4. Univariate Analysis (Distribution of a single variable)
# Choose a numerical column from your dataset to plot
numerical_col = df.select_dtypes(include=["int64", "float64"]).columns[0]

plt.figure(figsize=(8, 4))
sns.histplot(df[numerical_col], kde=True, bins=30, color="skyblue")
plt.title(f"Distribution of {numerical_col}")
plt.xlabel(numerical_col)
plt.ylabel("Frequency")
plt.show()

# 5. Bivariate/Multivariate Analysis (Relationships between variables)
# Correlation Heatmap for numeric features
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=["int64", "float64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()