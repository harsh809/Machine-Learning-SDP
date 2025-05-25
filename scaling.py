import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load input feature datasets

X_train_ros = pd.read_csv("X_train_ros.csv")
X_test = pd.read_csv("X_test.csv")

# Target labels
y_train_ros = pd.read_csv("y_train_ros.csv")
y_test = pd.read_csv("y_test.csv")

# Ensure consistent columns
columns = X_test.columns

# Helper function to scale and save
def scale_and_combine(X_train, y_train, X_test, y_test, prefix):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save individual scaled sets (optional)
    pd.DataFrame(X_train_scaled, columns=columns).to_csv(f"X_train_{prefix}_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=columns).to_csv(f"X_test_scaled_{prefix}.csv", index=False)

    # Combine with targets
    train_df = pd.DataFrame(X_train_scaled, columns=columns)
    train_df["IsBuggy"] = y_train
    test_df = pd.DataFrame(X_test_scaled, columns=columns)
    test_df["IsBuggy"] = y_test

    # Save final merged datasets
    train_df.to_csv(f"train_{prefix}_scaled.csv", index=False)
    test_df.to_csv(f"test_{prefix}_scaled.csv", index=False)

    print(f" {prefix.upper()} scaling and merging done.")


# Process all resampling types
scale_and_combine(X_train_ros, y_train_ros, X_test, y_test, "ros")

print(" All datasets scaled and combined with IsBuggy successfully.")
