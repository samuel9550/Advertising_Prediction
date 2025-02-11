import pandas as pd

# Load the dataset
file_path = "dataset/Advertising Budget and Sales.csv"  # Update the path if needed
df = pd.read_csv(file_path)

# Drop unnecessary index column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Check for missing values
df.dropna(inplace=True)

# Check for duplicates and remove them
df.drop_duplicates(inplace=True)

# Rename columns for consistency
df.rename(columns={
    'TV Ad Budget ($)': 'TV_Budget',
    'Radio Ad Budget ($)': 'Radio_Budget',
    'Newspaper Ad Budget ($)': 'Newspaper_Budget',
    'Sales ($)': 'Sales'
}, inplace=True)

# Save the cleaned dataset
df.to_csv("dataset/Cleaned_Advertising_Data.csv", index=False)

print("Data cleaning completed. Cleaned file saved as 'Cleaned_Advertising_Data.csv'.")
