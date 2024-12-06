import os
import sys
sys.path.insert(0, '/Users/adrianmolofsky/Downloads/CS229-Project/')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor

df = pd.read_csv("data/kalshi.csv")

df = df.drop(columns=[col for col in ['market', 'market_question', 'date'] if col in df.columns])

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(name, y_true, y_pred, additional_metrics=False):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    sharpe = np.divide(np.mean(y_pred), np.std(y_pred, ddof=1))
    
    print("{}:".format(name))
    print("  MSE: {}".format(mse))
    print("  RMSE: {}".format(rmse))
    print("  R²: {}".format(r2))
    if additional_metrics:
        print("  MAE: {}".format(mae))
        print("  MAPE: {}".format(mape))
        print("  EVS: {}".format(evs))
    print("  Sharpe Ratio: {}".format(sharpe))
    print()

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
evaluate_model("Linear Regression", y_test, y_pred_lr, additional_metrics=True)

gb = GradientBoostingRegressor(
    subsample=0.5,
    n_estimators=100,
    max_depth=7,
    learning_rate=0.001,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
evaluate_model("Gradient Boosting", y_test, y_pred_gb, additional_metrics=True)

cat = CatBoostRegressor(
    depth=4,
    iterations=121,
    learning_rate=0.0017066305219717408,
    random_state=42,
    verbose=0
)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)
evaluate_model("CatBoost", y_test, y_pred_cat, additional_metrics=True)
