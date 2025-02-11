import pandas as pd

# Load dataset
data = pd.read_csv("Advertising Budget and Sales.csv")

# Display first few rows
print("\nDataset Preview:")
print(data.head())

# Show column details
print("\nDataset Info:")
print(data.info())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove unnecessary column
data = data.drop(columns=["Unnamed: 0"], errors="ignore")

# Rename columns for easier access
data.columns = ["TV", "Radio", "Newspaper", "Sales"]

# Display cleaned dataset
print("\nCleaned Dataset Preview:")
print(data.head())

# Save the cleaned dataset
data.to_csv("Processed_Advertising_Data.csv", index=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Visualizing the distribution of Sales
plt.figure(figsize=(6, 4))
sns.histplot(data["Sales"], bins=20, kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# Scatter plots for relationships
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.scatterplot(x=data["TV"], y=data["Sales"])
plt.title("TV Budget vs Sales")

plt.subplot(1, 3, 2)
sns.scatterplot(x=data["Radio"], y=data["Sales"])
plt.title("Radio Budget vs Sales")

plt.subplot(1, 3, 3)
sns.scatterplot(x=data["Newspaper"], y=data["Sales"])
plt.title("Newspaper Budget vs Sales")

plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Selecting Features and Target Variable
X = data[['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']]
y = data['Sales ($)']

# Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Making Predictions
y_pred = model.predict(X_test)

# Evaluating the Model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Displaying Results
print("\nModel Performance:")
print(f"✅ R² Score: {r2:.4f} (higher is better)")
print(f"✅ RMSE: {rmse:.2f} (lower is better)")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = pd.read_csv("Advertising Budget and Sales.csv")

# Drop unnecessary columns
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

# ---- SECTION 2: MODEL TRAINING ----
print("\n🔹 Now Training Linear Regression Model...")

# Selecting Features and Target Variable
X = data[['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']]
y = data['Sales ($)']

# Splitting Data into Training and Testing Sets (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Making Predictions
y_pred = model.predict(X_test)

# Evaluating the Model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Displaying Results
print("\n📊 Model Performance:")
print(f"✅ R² Score: {r2:.4f} (higher is better)")
print(f"✅ RMSE: {rmse:.2f} (lower is better)")

# Save Model for Deployment
import pickle
with open("advertising_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("\n✅ Model saved as 'advertising_model.pkl' for Streamlit Deployment.")

# Rename columns to ensure correct format
data.columns = data.columns.str.strip()  # Removes unwanted spaces

# Print column names to check
print("\n🔹 Column Names in Dataset:", data.columns.tolist())

# Define Features and Target
X = data.iloc[:, 1:-1]  # Selects all columns except the first (ID) and last (Sales)
y = data.iloc[:, -1]    # Selects last column as the target (Sales)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = pd.read_csv("Advertising Budget and Sales.csv")

# Drop unnecessary columns
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

# Fix column names (strip spaces & standardize)
data.columns = data.columns.str.strip()

# Print column names to check
print("\n🔹 Column Names in Dataset:", data.columns.tolist())

# Selecting Features (All but last column) and Target (Last column)
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Model Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Display Metrics
print("\n📊 Model Performance:")
print(f"✅ R² Score: {r2:.4f} (higher is better)")
print(f"✅ RMSE: {rmse:.2f} (lower is better)")

# Save the Model for Deployment
import pickle
with open("advertising_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("\n✅ Model saved as 'advertising_model.pkl' for Streamlit Deployment.")

# Fix column names: Remove spaces and special characters
data.columns = data.columns.str.strip()  # Removes leading/trailing spaces
data.columns = data.columns.str.replace(r'[^\w\s]', '', regex=True)  # Remove special characters

# Print updated column names to verify
print("\n🔹 Cleaned Column Names:", data.columns.tolist())

# Select Features and Target with Updated Names
X = data[['TV Ad Budget', 'Radio Ad Budget', 'Newspaper Ad Budget']]  # Update names here
y = data['Sales']

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv("Advertising Budget and Sales.csv")

# Clean Column Names
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces
data.columns = data.columns.str.replace(r'[^\w\s]', '', regex=True)  # Remove special characters

# Print Cleaned Column Names to Verify
print("\n🔹 Cleaned Column Names:", data.columns.tolist())

# Drop unnecessary columns (e.g., index column if present)
if 'Unnamed' in data.columns[0]:  # Sometimes CSV files have an unnamed index column
    data.drop(data.columns[0], axis=1, inplace=True)

# Define Features (X) and Target (y)
X = data[['TV Ad Budget', 'Radio Ad Budget', 'Newspaper Ad Budget']]  # Update column names if needed
y = data['Sales']

# Split Data into Training & Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n✅ Data Preprocessing Completed!")
# Load Dataset
data = pd.read_csv("Advertising Budget and Sales.csv")

# Clean Column Names
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces
data.columns = data.columns.str.replace(r'[^\w\s]', '', regex=True)  # Remove special characters

# Print Cleaned Column Names to Verify
print("\n🔹 Cleaned Column Names:", data.columns.tolist())

# Drop unnecessary columns (e.g., index column if present)
if 'Unnamed' in data.columns[0]:  # Sometimes CSV files have an unnamed index column
    data.drop(data.columns[0], axis=1, inplace=True)

# Define Features (X) and Target (y)
X = data[['TV Ad Budget', 'Radio Ad Budget', 'Newspaper Ad Budget']]  # Update column names if needed
y = data['Sales']

# Split Data into Training & Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n✅ Data Preprocessing Completed!")
