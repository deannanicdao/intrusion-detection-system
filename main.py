# main.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Load the UNSW-NB15 dataset
# Replace the file paths with the actual paths to your training and testing CSV files
train_data = pd.read_csv("data/UNSW_NB15_training-set.csv")
test_data = pd.read_csv("data/UNSW_NB15_testing-set.csv")

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
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Perform cross-validation (check for overfitting - asses the model's generalization ability)
# Cross-validation to evaluate the model's performance
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

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


