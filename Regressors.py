import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform


graph_data = pd.read_csv(r'D:/1_data/GraphData/graph_data.csv').drop(['Unnamed: 0', 'owner', 'repo', 'cloneURL', 'Repository', 'Repository URL',
       'Total nloc', 'Avg.NLOC', 'Avg.token', 'Fun Cnt',
       'file threshold cnt', 'Fun Rt', 'nloc Rt'], axis=1)
graph_encodings = pd.read_csv(r'D:/1_data/GraphData/encodings.csv', index_col=0).dropna()
all_metrics = pd.merge(graph_data, graph_encodings, left_index=True, right_on='id')
all_metrics.to_csv(r'D:/1_data/GraphData/all_data.csv')

# Load the data
df = pd.read_csv(r'D:/1_data/GraphData/all_data.csv')
print(df.columns)

# ,
#
# Selecting features and target variables
features = ['stars', 'dateCreated', 'datePushed', 'numCommits', 'openIssues', 'closedIssues', 'totalIssues', 'totalAdditions', 'totalDeletions', 'fileCount', '1', '2', '3', '4', '5', '6', '7', '8']  # Define your feature columns
target_variable = 'AvgCCN'  # Define your target column

X = df[features]
X = normalize(X, axis=0)
y = df[target_variable]

# Splitting the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forests - Hyperparameters
rf_param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None] + list(range(5, 30, 5))
}

# Randomized Search for Random Forests
rf_random_search = RandomizedSearchCV(RandomForestRegressor(), param_distributions=rf_param_dist, n_iter=10, cv=3, random_state=42)
rf_random_search.fit(X_train, y_train)
rf_model = rf_random_search.best_estimator_
rf_predictions = rf_model.predict(X_test)
rf_mse_valid = mean_squared_error(y_test, rf_predictions)
rf_r2_valid = r2_score(y_test, rf_predictions)
rf_mape_valid = np.mean(np.abs((rf_predictions - y_test) / y_test)) * 100
print("\nResults for Random Forest on Validation Set:")
print(f"Mean Squared Error: {rf_mse_valid:.2f}, R^2 Score: {rf_r2_valid:.2f}, MAPE: {rf_mape_valid:.2f}%")

# K-Nearest Neighbors - Hyperparameters
knn_param_dist = {
    'n_neighbors': randint(3, 10),
}

# Randomized Search for KNN
knn_random_search = RandomizedSearchCV(KNeighborsRegressor(), param_distributions=knn_param_dist, n_iter=10, cv=3, random_state=42)
knn_random_search.fit(X_train, y_train)
knn_model = knn_random_search.best_estimator_
knn_predictions = knn_model.predict(X_test)
knn_mse_valid = mean_squared_error(y_test, knn_predictions)
knn_r2_valid = r2_score(y_test, knn_predictions)
knn_mape_valid = np.mean(np.abs((knn_predictions - y_test) / y_test)) * 100
print("\nResults for KNN on Validation Set:")
print(f"Mean Squared Error: {knn_mse_valid:.2f}, R^2 Score: {knn_r2_valid:.2f}, MAPE: {knn_mape_valid:.2f}%")

# Lasso Regression - Hyperparameters
lasso_param_dist = {
    'alpha': uniform(0.1, 10),
}

# Randomized Search for Lasso Regression
lasso_random_search = RandomizedSearchCV(Lasso(), param_distributions=lasso_param_dist, n_iter=10, cv=3, random_state=42)
lasso_random_search.fit(X_train, y_train)
lasso_model = lasso_random_search.best_estimator_
lasso_predictions = lasso_model.predict(X_test)
lasso_mse_valid = mean_squared_error(y_test, lasso_predictions)
lasso_r2_valid = r2_score(y_test, lasso_predictions)
lasso_mape_valid = np.mean(np.abs((lasso_predictions - y_test) / y_test)) * 100
print("\nResults for Lasso Regression on Validation Set:")
print(f"Mean Squared Error: {lasso_mse_valid:.2f}, R^2 Score: {lasso_r2_valid:.2f}, MAPE: {lasso_mape_valid:.2f}%\n")

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_test)

# Define hyperparameters range for random search
param_dist = {
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5)
}

# Perform randomized search
num_rounds = 100  # Adjust as needed
best_score = float('inf')
best_params = None

for _ in range(10):  # Adjust the number of iterations as needed
    params = {key: dist.rvs() for key, dist in param_dist.items()}
    params['objective'] = 'reg:squarederror'
    params['eval_metric'] = 'rmse'

    model = xgb.train(params, dtrain, num_rounds, evals=[(dvalid, 'validation')], early_stopping_rounds=10, verbose_eval=False)
    predictions = model.predict(dvalid)
    mse = mean_squared_error(y_valid, predictions)

    if mse < best_score:
        best_score = mse
        best_params = params

# Train the best model
final_model = xgb.train(best_params, dtrain, num_rounds)

# Predictions on the test set
predictions = final_model.predict(dtest)

# Evaluate the final model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mape = np.mean(np.abs((predictions - y_test) / y_test)) * 100
print("\nResults for XGBoost Regression on Validation Set:")
print(f"Mean Squared Error: {mse:.2f}, R^2 Score: {r2:.2f}, MAPE: {mape:.2f}%")
