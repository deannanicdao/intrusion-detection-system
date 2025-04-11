# main.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Replace 4 with the number of cores you want to use

# Load the UNSW-NB15 dataset
# Replace the file paths with the actual paths to your training and testing CSV files
train_data = pd.read_csv("data/UNSW_NB15_training-set.csv")
test_data = pd.read_csv("data/UNSW_NB15_testing-set.csv")
testing_columns = test_data.columns
print(testing_columns)

# Combine training and testing data for preprocessing
data = pd.concat([train_data, test_data], ignore_index=True)

# Preprocess data
# Drop irrelevant columns (e.g., 'id') and handle categorical features
data = pd.get_dummies(data, drop_first=True)  # One-hot encode categorical features

# Preprocess training data
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_train = pd.get_dummies(X_train, drop_first=True)

# Preprocess test data
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns in X_train and X_test to ensure they have the same features
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Identify categorical columns
categorical_columns = X_train.select_dtypes(include=['object']).columns

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    label_encoders[col] = le

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Combine resampled features and target into a new DataFrame (optional, for saving or inspection)
resampled_data = pd.concat([pd.DataFrame(X_train_resampled, columns=X_train.columns), 
                            pd.DataFrame(y_train_resampled, columns=['label'])], axis=1)

# Save the resampled dataset (optional)
resampled_data.to_csv("data/UNSW_NB15_resampled.csv", index=False)

# Split data into features (X) and target (y)
X = data.drop('label', axis=1)  # Features
y = data['label']  # Target (0 for normal, 1 for malicious)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
# Initialize the Random Forest model with best hyperparameters (using GridSearchCV or RandomizedSearchCV to find the best parameters)
model = RandomForestClassifier(
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)

# Perform cross-validation (check for overfitting - asses the model's generalization ability)
# Cross-validation to evaluate the model's performance
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Generate the classification report as a dictionary
report = classification_report(y_test, y_pred, output_dict=True)

# Convert the report to a DataFrame for visualization
report_df = pd.DataFrame(report).transpose()

# Plot precision, recall, and F1-score for each class
metrics = ['precision', 'recall', 'f1-score']
report_df = report_df.loc[['0', '1'], metrics]  # Select only class 0 and 1

report_df.plot(kind='bar', figsize=(8, 6))
plt.title("Classification Metrics by Class")
plt.xlabel("Class")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("classification_metrics.png")
plt.show()

# Get feature importances (which are most important for the model's predictions)
feature_importances = model.feature_importances_
features = X.columns

# Sort and select top 10 important features
indices = np.argsort(feature_importances)[::-1]
important_features = features[indices][:10]  # Keep only the top 10 features
X = X[important_features]

# Plot feature importances for the top 10 features
plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Top 10)")
plt.bar(range(len(important_features)), feature_importances[indices][:10], align="center")
plt.xticks(range(len(important_features)), important_features, rotation=90)
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()


