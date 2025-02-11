import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = "dataset/Cleaned_Advertising_Data.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Display first few rows
print("First 5 Rows of the Dataset:")
print(df.head())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Check correlation between features
print("\nCorrelation Matrix:")
print(df.corr())

# Visualizing correlation using heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for feature relationships
sns.pairplot(df)
plt.show()

# Histogram for Sales Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Sales'], bins=20, kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales ($)")
plt.ylabel("Frequency")
plt.show()

# Relationship between Ad Budgets and Sales
plt.figure(figsize=(12, 4))

# TV Budget vs Sales
plt.subplot(1, 3, 1)
sns.scatterplot(x=df['TV_Budget'], y=df['Sales'])
plt.title("TV Budget vs Sales")

# Radio Budget vs Sales
plt.subplot(1, 3, 2)
sns.scatterplot(x=df['Radio_Budget'], y=df['Sales'])
plt.title("Radio Budget vs Sales")

# Newspaper Budget vs Sales
plt.subplot(1, 3, 3)
sns.scatterplot(x=df['Newspaper_Budget'], y=df['Sales'])
plt.title("Newspaper Budget vs Sales")

plt.tight_layout()
plt.show()

# Print insights based on correlation
correlation_matrix = df.corr()
strongest_correlation = correlation_matrix["Sales"].drop("Sales").idxmax()
print(f"\nStrongest correlation with Sales is: {strongest_correlation}")

