import pandas as pd

# 1. Load the dataset
file_path = "parkinsons_updrs.data"
df = pd.read_csv(file_path)

# 2. Display basic dimensions
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total Rows (Recordings): {df.shape[0]}")
print(f"Total Columns (Features): {df.shape[1]}")

# 3. Inspect unique subjects
unique_subjects = df['subject#'].nunique()
print(f"Number of Unique Patients (subject#): {unique_subjects}")

# 4. Display column details and missing values
print("\n" + "=" * 60)
print("COLUMN DETAILS & MISSING VALUES")
print("=" * 60)
info_df = pd.DataFrame({
    'Data Type': df.dtypes,
    'Missing Values': df.isnull().sum()
})
print(info_df)

# 5. Display the first 5 rows
print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())