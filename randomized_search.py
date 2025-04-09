from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from main import X_train, y_train

# Define the parameter distribution
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None]
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Initialize RandomizedSearchCV
# Note: jobs can be set to -1 to use all processors, or a specific number of processors
# to limit the number of parallel jobs
# n_iter is the number of different combinations to try
# cv is the number of cross-validation folds, 5 here means each combination of hyperparameters 
# will be evaluated 5 times (once for each fold) until all combinations in the parameter grid have been evaluated
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', verbose=2, random_state=42, n_jobs=2)

# Fit RandomizedSearchCV to the data
random_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation Accuracy:", random_search.best_score_)