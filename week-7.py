# Data Analysis Assignment

# Using Pandas, Matplotlib, and Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# --------------------------
# Task 1: Load & Explore Data
# --------------------------

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Display first 5 rows
print("First rows of the dataset:")
print(df.head())

# Data types and missing values
print("\nDataset Info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

# Clean dataset (no missing values here, but we'll demonstrate filling)
df.fillna(df.mean(numeric_only=True), inplace=True)

# --------------------------
# Task 2: Basic Data Analysis
# --------------------------

# Compute statistics
print("\nStatistical Summary:")
print(df.describe())

# Grouping: mean petal length per species
grouped = df.groupby("target")["petal length (cm)"].mean()
print("\nAverage petal length per species:")
print(grouped)

# --------------------------
# Task 3: Data Visualizations
# --------------------------

# 1. Line chart - sepal length trend (first 30 rows as "time-series")
plt.figure(figsize=(8,5))
plt.plot(df.index[:30], df["sepal length (cm)"][:30], marker="o")
plt.title("Sepal Length Trend (First 30 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.grid()
plt.show()

# 2. Bar chart - average sepal width per species
plt.figure(figsize=(8,5))
sns.barplot(x="target", y="sepal width (cm)", data=df, estimator="mean")
plt.title("Average Sepal Width per Species")
plt.xlabel("Species")
plt.ylabel("Average Sepal Width (cm)")
plt.show()

# 3. Histogram - distribution of petal length
plt.figure(figsize=(8,5))
plt.hist(df["petal length (cm)"], bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot - Sepal Length vs Petal Length
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="target", data=df, palette="Set1")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

