# main.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
