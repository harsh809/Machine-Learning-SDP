import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Step 1: Load the feature-selected dataset
df = pd.read_csv("/content/feature_selected_dataset.csv")

# Step 2: Separate features and target
X = df.drop(columns=['IsBuggy'])
y = df['IsBuggy']

# Step 3: Stratified Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Original class distribution in y_train:\n", y_train.value_counts())
print("Original class distribution in y_test:\n", y_test.value_counts())
print("Original class distribution in X_test:\n", X_test.value_counts())
print("Original class distribution in X_train:\n", X_train.value_counts())

# Step 4: Initialize resamplers
ros = RandomOverSampler(random_state=42)

# Step 5: Apply resampling on training data
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

# Step 6: Reset indices to avoid mismatch during modeling
datasets = [
    (X_train, y_train, "original"),
    (X_train_ros, y_train_ros, "ros"),
]

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Step 7: Save all datasets
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

for X_res, y_res, name in datasets:
    X_res = X_res.reset_index(drop=True)
    y_res = y_res.reset_index(drop=True)
    X_res.to_csv(f"X_train_{name}.csv", index=False)
    y_res.to_csv(f"y_train_{name}.csv", index=False)

print("Train/test split and resampled datasets (ROS) saved.")
