import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

train_data = pd.read_csv('train_df.csv')
test_data = pd.read_csv('test_df.csv')

X_train, X_validation, y_train, y_validation = train_test_split(
    train_data.iloc[:, 1:80],  # Признаки
    train_data['target'],       # Целевая переменная
    test_size=0.2,
    random_state=42
)

regression_model = xgb.XGBRegressor(objective='reg:squarederror')
regression_model.fit(X_train, y_train)

validation_predictions = regression_model.predict(X_validation)
ndcg_validation = ndcg_score(np.array([y_validation]), np.array([validation_predictions]))
print(f"NDCG train: {ndcg_validation}")

test_predictions = regression_model.predict(test_data.iloc[:, 1:80])

ndcg_test = ndcg_score(np.array([test_data['target']]), np.array([test_predictions]))
print(f"NDCG test: {ndcg_test}")
