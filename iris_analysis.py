
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Clean dataset (no missing values in this case)
cleaned_df = df.dropna()

# Basic statistics
print("\nBasic Statistics:")
print(cleaned_df.describe())

# Group by species and compute the mean
grouped_means = cleaned_df.groupby('target').mean()
print("\nMean values grouped by target:")
print(grouped_means)

# Rename target values for clarity
target_names = iris.target_names
cleaned_df['species'] = cleaned_df['target'].apply(lambda x: target_names[x])

# Visualization setup
plt.figure(figsize=(16, 12))
plt.suptitle("Iris Dataset Visualizations", fontsize=20)

# Line chart: simulate time-series by plotting row index
plt.subplot(2, 2, 1)
for species in cleaned_df['species'].unique():
    species_data = cleaned_df[cleaned_df['species'] == species]
    plt.plot(species_data.index, species_data['sepal length (cm)'], label=species)
plt.title('Sepal Length Over Index')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()

# Bar chart: average petal length per species
plt.subplot(2, 2, 2)
sns.barplot(data=cleaned_df, x='species', y='petal length (cm)')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')

# Histogram: distribution of sepal width
plt.subplot(2, 2, 3)
plt.hist(cleaned_df['sepal width (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')

# Scatter plot: sepal length vs. petal length
plt.subplot(2, 2, 4)
sns.scatterplot(data=cleaned_df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
