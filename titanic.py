import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "Titanic-Dataset.csv"  
df = pd.read_csv(file_path)

# Step 1: Check for missing values in "Age" and "Fare"
missing_age = df["Age"].isnull().sum()
missing_fare = df["Fare"].isnull().sum()

print(f"Missing values in 'Age': {missing_age}")
print(f"Missing values in 'Fare': {missing_fare}")

# Step 2: Plot the distribution of "Age" and "Fare"
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df["Age"], bins=30, kde=True, color="blue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.histplot(df["Fare"], bins=30, kde=True, color="green")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Step 3: Impute missing values for "Age" using the mean method
age_mean = df["Age"].mean()
df["Age"] = df["Age"].fillna(age_mean)  # Fixed inplace warning

print(f"Missing values in 'Age' after imputation: {df['Age'].isnull().sum()}")

# Step 4: Compute the Z-score for "Age"
df["Age_zscore"] = stats.zscore(df["Age"])

# Step 5: Count the number of outliers where Z > 3
outliers_count = (df["Age_zscore"] > 3).sum()
print(f"Number of outliers in 'Age' (Z > 3): {outliers_count}")

# Step 6: Compute the probability P(Age < 20) assuming normal distribution
age_std = df["Age"].std()
probability_age_less_20 = stats.norm.cdf(20, loc=age_mean, scale=age_std)

print(f"Probability that 'Age' < 20: {probability_age_less_20:.4f} (or {probability_age_less_20 * 100:.2f}%)")

