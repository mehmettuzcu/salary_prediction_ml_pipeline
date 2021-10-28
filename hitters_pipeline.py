
########################### End-to-End Hitters Machine Learning Pipeline II ###########################

import joblib
import os
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate,\
    validation_curve, train_test_split

from helpers.eda import *
from helpers.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


#################### Functions ############################

# Data Preprocessing & Feature Engineering

def hitters_data_pred(dataframe):
    dataframe.dropna(inplace=True)
    dataframe["HITS_SUCCESS"] = (dataframe["AtBat"] - dataframe["Hits"])

    dataframe["NEW_ATBAT_CATBAT_RATE"] = dataframe["AtBat"] / dataframe["CAtBat"]

    dataframe["NEW_RUN_RATE"] = dataframe["Runs"] / dataframe["CRuns"]

    dataframe["NEW_RBI_RATE"] = dataframe["RBI"] / (dataframe["CRBI"] + 0.00001)

    dataframe["NEW_HITS_RATE"] = dataframe["Hits"] / dataframe["CHits"]

    dataframe["NEW_WALKS_RATE"] = dataframe["Walks"] / dataframe["CWalks"]

    dataframe["NEW_WALKS_RATE"] = dataframe["CHits"] / dataframe["CAtBat"]

    dataframe["NEW_CRUNS_RATE"] = dataframe["CRuns"] / dataframe["Years"]

    dataframe["NEW_CHITS_RATE"] = dataframe["CHits"] / dataframe["Years"]

    dataframe["NEW_TOTAL_BASES"] = ((dataframe["CHits"] * 2) + (4 * dataframe["CHmRun"]))

    dataframe["NEW_SLUGGIN_PERCENTAGE"] = dataframe["NEW_TOTAL_BASES"] / dataframe["CAtBat"]

    dataframe["NEW_ISOLETED_POWER"] = dataframe["NEW_SLUGGIN_PERCENTAGE"] - dataframe["NEW_WALKS_RATE"]

    dataframe["NEW_TRIPLW_CROWN"] = (dataframe["CHmRun"] * 0.4) + (dataframe["CRBI"] * 0.25) + (dataframe["NEW_WALKS_RATE"] * 0.35)

    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float] and dataframe[col].nunique() == 2]

    len(binary_cols)

    for col in binary_cols:
        label_encoder(dataframe, col)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)
    dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)

    ################## Outlier Review ##################

    num_cols.remove("Salary")
    for col in num_cols:
        check_outlier(dataframe, col, 0.1, 0.9)
        if check_outlier(dataframe, col):
            replace_with_thresholds(dataframe, col)
    y = dataframe["Salary"]
    X = dataframe.drop(["Salary"], axis=1)
    return X, y


########################## Base Models ############################

def base_models(X, y, scoring="RMSE"):
    print("Base Models....")
    models = [('LR', LinearRegression()),
              ("Ridge", Ridge()),
              ("Lasso", Lasso()),
              ("ElasticNet", ElasticNet()),
              ('KNN', KNeighborsRegressor()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              ('SVR', SVR()),
              ('GBM', GradientBoostingRegressor()),
              ("XGBoost", XGBRegressor(objective='reg:squarederror')),
              ("LightGBM", LGBMRegressor()),
              # ("CatBoost", CatBoostRegressor(verbose=False))
              ]

    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"{scoring}: {round(rmse, 4)} ({name}) ")


#################### Automated Hyperparameter Optimization ###################################

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [6, 7, 8, "auto"],
             "min_samples_split": [14, 15, 16],
             "n_estimators": [175, 178, 180, 185]}

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 16, 20],
                  "n_estimators": [100, 200, 300, 500],
                  "colsample_bytree": [0.5, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
                   "n_estimators": [300, 500, 1500],
                   "colsample_bytree": [0.5, 0.7, 1]}

regressors = [("CART", DecisionTreeRegressor(), cart_params),
              ("RF", RandomForestRegressor(), rf_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="neg_mean_squared_error"):
    print("Hyperparameter Optimization")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=cv, scoring=scoring)))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

        gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

        final_model = regressor.set_params(**gs_best.best_params_)
        rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        best_models[name] = final_model
    return best_models
########################## Stacking & Ensemble Learning ############################

def voting_regressorr(best_models, X, y):
    print("Voting Regressor")
    voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('LightGBM', best_models["LightGBM"])]).fit(X, y)

    cv_results = np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {cv_results}")

    return voting_reg

##################### Pipeline Main Function ###########################
def main():
    df = pd.read_csv("C:/Users/Tuzcu/Desktop/DSMLBC/datasets/hitters.csv")
    X, y = hitters_data_pred(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_reg = voting_regressorr(best_models, X, y)
    os.chdir("/Users/Tuzcu/Desktop/DSMLBC/")
    joblib.dump(voting_reg, "voting_reg_hitters.pkl")
    print("Voting_reg has been created")
    return voting_reg

if __name__ == "__main__":
    main()
