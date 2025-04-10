# main.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the UNSW-NB15 dataset
# Replace the file paths with the actual paths to your training and testing CSV files
train_data = pd.read_csv("data/UNSW_NB15_training-set.csv")
test_data = pd.read_csv("data/UNSW_NB15_testing-set.csv")
testing_columns = test_data.columns
print(testing_columns)

# Load the unseen test dataset
unseen_test_data = pd.read_csv("data/UNSW-NB15_1.csv", low_memory=False)
unseen_test_data_reordered = unseen_test_data.reindex(columns=testing_columns)
unseen_test_data_reordered.to_csv("data/UNSW_NB15_unseen-test.csv", index=False)

# Check the columns of the unseen test data to ensure they match the training data
print("Unseen Test Data Columns:")
print(unseen_test_data.columns)

# Drop irrelevant or high-cardinality columns
unseen_test_data = unseen_test_data.drop(['id'], axis=1, errors='ignore')

# Define categorical columns (update based on actual column names)
categorical_columns = ['proto', 'service', 'state']  # Replace with actual column names
label_encoders = {}

# Apply LabelEncoder to existing columns
for col in categorical_columns:
    if col in unseen_test_data.columns:
        le = LabelEncoder()
        unseen_test_data[col] = le.fit_transform(unseen_test_data[col].astype(str))
        label_encoders[col] = le
    else:
        print(f"Column '{col}' not found in unseen_test_data. Skipping...")

# Reorder columns to match the testing dataset
unseen_test_data = unseen_test_data.reindex(columns=testing_columns, fill_value=0)

# Split unseen test data into features and target
X_unseen = unseen_test_data.drop('label', axis=1)
y_unseen = unseen_test_data['label']

# Combine training and testing data for preprocessing
data = pd.concat([train_data, test_data], ignore_index=True)

# Preprocess data
# Drop irrelevant columns (e.g., 'id') and handle categorical features
data = data.drop(['id'], axis=1)  # Drop the 'id' column
data = pd.get_dummies(data, drop_first=True)  # One-hot encode categorical features

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
model.fit(X_train, y_train)

# Perform cross-validation (check for overfitting - asses the model's generalization ability)
# Cross-validation to evaluate the model's performance
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Make predictions on the unseen test data
y_unseen_pred = model.predict(X_unseen)

# Evaluate the model's performance on the unseen test data
print("Evaluation on Unseen Test Data:")
print(classification_report(y_unseen, y_unseen_pred))

# Save predictions to a CSV file
unseen_test_data['predicted_label'] = y_unseen_pred
unseen_test_data.to_csv("data/UNSW_NB15_unseen-test-predictions.csv", index=False)

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


