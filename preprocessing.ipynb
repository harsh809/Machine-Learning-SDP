import pandas as pd

# Step 1: Load the dataset
file_path = '/content/sample_data/REMOVEDFinal_dataset.xlsx'
data = pd.ExcelFile(file_path)
df = data.parse(data.sheet_names[0])  # Load the first sheet

# Step 2: Drop non-informative or identifier columns
if 'Kind' in df.columns and 'Name' in df.columns:
    df = df.drop(columns=['Kind', 'Name'])

# Step 3: Drop columns with more than 50% missing values
missing_values = df.isnull().sum()
columns_to_drop = missing_values[missing_values > 0.5 * len(df)].index
df = df.drop(columns=columns_to_drop)

# Step 4: Split dataset into buggy and non-buggy
buggy_df = df[df['IsBuggy'] == 1].copy()
non_buggy_df = df[df['IsBuggy'] == 0].copy()

# Step 5: Clean only non-buggy rows → remove rows with >40% zero values
feature_cols = [col for col in non_buggy_df.columns if col != 'IsBuggy']
zero_threshold = 0.4 * len(feature_cols)
non_buggy_df = non_buggy_df[(non_buggy_df[feature_cols] == 0).sum(axis=1) < zero_threshold]

# Step 6: Drop duplicates only from non-buggy data
non_buggy_df = non_buggy_df.drop_duplicates()

# Step 7: Combine buggy + cleaned non-buggy
df_cleaned = pd.concat([buggy_df, non_buggy_df], ignore_index=True)

# Step 8: Fill missing values using column-wise mean (only for numeric columns)
numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())

# Step 9: Shuffle the dataset
df_cleaned = df_cleaned.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 10: Final shape and class balance
print("Final dataset shape:", df_cleaned.shape)
print("Buggy class distribution:\n", df_cleaned['IsBuggy'].value_counts())

# Step 11: Save cleaned dataset
df_cleaned.to_csv("cleaned_stable_dataset.csv", index=False)
print("Cleaned dataset saved as 'cleaned_stable_dataset.csv'.")
