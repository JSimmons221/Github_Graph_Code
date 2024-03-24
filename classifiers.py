import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss, confusion_matrix
from scipy.stats import randint, uniform
from sklearn.linear_model import Lasso
import seaborn as sns
import matplotlib.pyplot as plt

# graph_data = pd.read_csv(r'D:/1_data/GraphData/graph_data.csv').drop(['Unnamed: 0', 'cloneURL', 'Repository',
#        'Repository URL', 'Total nloc', 'Avg.NLOC', 'Avg.token', 'Fun Cnt', 'file threshold cnt', 'Fun Rt', 'nloc Rt',
#        'AvgCCN', 'Halstead Volume', 'Maintainability Index'], axis=1)
# graph_encodings = pd.read_csv(r'D:/1_data/GraphData/encodings.csv', index_col=0).dropna()
# data_metrics = pd.merge(graph_data, graph_encodings, left_index=True, right_on='id')
# graph_classes = pd.read_csv(r'D:/1_data/GraphData/repo_status.csv')
# classifier_data_unclean = pd.merge(data_metrics, graph_classes, left_on=['owner', 'repo'], right_on=['Owner', 'Repo'])
# classifier_data = classifier_data_unclean[classifier_data_unclean['Status'].isin(['alive', 'dead'])].drop(['owner', 'repo', 'Owner', 'Repo'], axis=1)
# classifier_data = classifier_data.replace(to_replace='alive', value=1).replace(to_replace='dead', value=0)
# classifier_data.to_csv(r'D:/1_data/GraphData/classifier_data.csv', index=False)

classifier_data = pd.read_csv(r'D:/1_data/GraphData/classifier_data.csv')
print(classifier_data.columns)

#
features = ['stars', 'dateCreated', 'datePushed', 'numCommits', 'openIssues',
       'closedIssues', 'totalIssues', 'totalAdditions', 'totalDeletions',
       'fileCount', '1', '2', '3', '4', '5', '6', '7', '8']  # Define your feature columns
target_variable = 'Status'

X = classifier_data[features]
X = normalize(X, axis=0)
y = classifier_data[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

rf_param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None] + list(range(5, 30, 5))
}

rf_random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_param_dist, n_iter=10, cv=3, random_state=42)
rf_random_search.fit(X_train, y_train)
rf_model = rf_random_search.best_estimator_
rf_predictions = rf_model.predict(X_test)
rf_loss_val = log_loss(y_test, rf_predictions)
rf_cm_val = confusion_matrix(y_test, rf_predictions)

print("Random Forests")
print(rf_loss_val)
sns.heatmap(rf_cm_val, annot=True)
plt.show()
print()

knn_param_dist = {
    'n_neighbors': randint(3, 10),
}

# Randomized Search for KNN
knn_random_search = RandomizedSearchCV(KNeighborsClassifier(), param_distributions=knn_param_dist, n_iter=10, cv=3, random_state=42)
knn_random_search.fit(X_train, y_train)
knn_model = knn_random_search.best_estimator_
knn_predictions = knn_model.predict(X_test)
knn_loss_val = log_loss(y_test, knn_predictions)
knn_cm_val = confusion_matrix(y_test, knn_predictions)

print("K Nearest Neighbors")
print(knn_loss_val)
sns.heatmap(knn_cm_val, annot=True)
plt.show()
print()

# Lasso Regression - Hyperparameters
lasso_param_dist = {
    'alpha': uniform(0.1, 10),
}

# Randomized Search for Lasso Regression
lasso_random_search = RandomizedSearchCV(Lasso(), param_distributions=lasso_param_dist, n_iter=10, cv=3, random_state=42)
lasso_random_search.fit(X_train, y_train)
lasso_model = lasso_random_search.best_estimator_
lasso_predictions = lasso_model.predict(X_test)
lasso_loss_val = log_loss(y_test, lasso_predictions.round())
lasso_cm_val = confusion_matrix(y_test, lasso_predictions.round())

print('Lasso')
print(lasso_loss_val)
sns.heatmap(lasso_cm_val, annot=True)
plt.show()
print()


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

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
    loss = log_loss(y_valid, predictions.round())

    if loss < best_score:
        best_score = loss
        best_params = params

final_model = xgb.train(best_params, dtrain, num_rounds)
predictions = final_model.predict(dtest)
xg_loss_val = log_loss(y_test, predictions.round())
xg_cm_val = confusion_matrix(y_test, predictions.round())

print('XGBoost')
print(xg_loss_val)
sns.heatmap(xg_cm_val, annot=True)
plt.show()
