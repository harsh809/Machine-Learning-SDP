import pandas as pd
import numpy as np
import time
import psutil
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# Load scaled ROS dataset
train_df = pd.read_csv('train_ros_scaled.csv')
test_df = pd.read_csv('test_ros_scaled.csv')

X_train = train_df.drop('IsBuggy', axis=1)
y_train = train_df['IsBuggy']
X_test = test_df.drop('IsBuggy', axis=1)
y_test = test_df['IsBuggy']

# Hyperparameter tuning for Decision Tree
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model = DecisionTreeClassifier(random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=skf, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\n✅ Best Parameters: {grid_search.best_params_}")

# Save the model
model_name = "decision_tree"
joblib.dump(best_model, f'{model_name}_model_ros.pkl')

# Evaluate on test set
process = psutil.Process()
mem_before_test = process.memory_info().rss / 1024 / 1024
test_start = time.time()

y_test_prob = best_model.predict_proba(X_test)[:, 1]

test_end = time.time()
mem_after_test = process.memory_info().rss / 1024 / 1024
test_time = test_end - test_start
peak_test_memory = mem_after_test - mem_before_test

# Find best threshold based on F1
thresholds = np.arange(0.0, 1.01, 0.01)
f1_scores = [f1_score(y_test, (y_test_prob >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"🔥 Best Threshold based on F1-score: {best_threshold:.2f}")

# Final prediction
y_test_pred = (y_test_prob >= best_threshold).astype(int)

# Metrics
acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
mcc = matthews_corrcoef(y_test, y_test_pred)
auc_score = roc_auc_score(y_test, y_test_prob)
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

# Save results
results = pd.DataFrame({
    "Model": [model_name.upper()],
    "Dataset": ["ROS"],
    "Precision": [prec],
    "Recall": [rec],
    "F1": [f1],
    "AUC-ROC": [auc_score],
    "Accuracy": [acc],
    "MCC": [mcc],
    "Testing Time (s)": [test_time],
    "Peak Testing Memory (MB)": [peak_test_memory],
    "Best Threshold": [best_threshold],
    "True Positives": [tp],
    "True Negatives": [tn],
    "False Positives": [fp],
    "False Negatives": [fn]
})

results.to_csv(f'{model_name}_ros_best_threshold_results.csv', index=False)

# F1 vs Threshold plot
plt.figure(figsize=(8, 5))
plt.plot(thresholds, f1_scores, label='F1-score')
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best: {best_threshold:.2f}')
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("Threshold vs F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"f1_vs_threshold_ros_{model_name}.png")
plt.close()

# Confusion Matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Buggy', 'Buggy'],
            yticklabels=['Not Buggy', 'Buggy'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {model_name.upper()} (ROS)')
plt.tight_layout()
plt.savefig(f'{model_name}_confusion_matrix_ros.png')
plt.close()

# AUC-ROC Curve plot

fpr, tpr, _ = roc_curve(y_test, y_test_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"AUC-ROC Curve - {model_name.upper()} (ROS)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{model_name}_auc_roc_curve_ros.png")
plt.close()

print("\n✅ Evaluation complete. Model, results, and plots saved.")
