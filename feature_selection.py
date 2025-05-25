import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Step 1: Load the cleaned dataset
df = pd.read_csv("/content/cleaned_stable_dataset.csv")

# Step 2: Separate features and target
X = df.drop(columns=['IsBuggy'])
y = df['IsBuggy']

# Step 3: Remove low-variance features (less informative)
var_thresh = VarianceThreshold(threshold=0.01)
X_var = var_thresh.fit_transform(X)

# Get names of features that remain after variance threshold
selected_columns = X.columns[var_thresh.get_support()]
X_var_df = pd.DataFrame(X_var, columns=selected_columns)

# Step 4: Feature importance using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_var_df, y)

# Step 5: Select top 15 important features
importances = pd.Series(rf.feature_importances_, index=X_var_df.columns)
top_features = importances.sort_values(ascending=False).head(15).index.tolist()

# Step 6: Final selected dataset
X_selected = X_var_df[top_features]
df_selected = X_selected.copy()
df_selected['IsBuggy'] = y.reset_index(drop=True)

# Step 7: Save the final dataset with selected features
df_selected.to_csv("feature_selected_dataset.csv", index=False)
print("Feature-selected dataset saved as 'feature_selected_dataset.csv'")

# Step 8: Log the top 15 selected features
print("Top 15 Selected Features:\n", top_features)

# Step 9: Plot the feature importances
importances.sort_values(ascending=False).head(15).plot(kind='barh', figsize=(8,6))
plt.title("Top 15 Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance_plot.png")
plt.show()
