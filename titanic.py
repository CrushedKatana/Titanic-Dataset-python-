import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "/mnt/data/Titanic-Dataset.csv"
df = pd.read_csv(file_path)

# Display basic info to understand the structure
df.info(), df.head()
# Plot the distribution of "Age" and "Fare"
plt.figure(figsize=(12, 5))

# Age Distribution
plt.subplot(1, 2, 1)
sns.histplot(df["Age"], bins=30, kde=True, color="blue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

# Fare Distribution
plt.subplot(1, 2, 2)
sns.histplot(df["Fare"], bins=30, kde=True, color="green")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Impute missing values in "Age" with the mean
age_mean = df["Age"].mean()
df["Age"].fillna(age_mean, inplace=True)

# Confirm there are no missing values left in "Age"
df["Age"].isnull().sum()

# Compute z-score for Age
df["Age_zscore"] = stats.zscore(df["Age"])

# Count the number of outliers (Z > 3)
outliers_count = (df["Age_zscore"] > 3).sum()
outliers_count

# Calculate probability P(Age < 20) assuming normal distribution
age_mean = df["Age"].mean()
age_std = df["Age"].std()

probability_age_less_20 = stats.norm.cdf(20, loc=age_mean, scale=age_std)
probability_age_less_20
