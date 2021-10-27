#############################################
# Salary Prediction with Machine Learning
#############################################

########################## Libraries and Utilities ##########################

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

#####################  1. Exploratory Data Analysis ###########################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Types #####################")
    print(dataframe.info())
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #######################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)


def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots(1, 2)
        ax = np.reshape(ax, (1, 2))
        ax[0, 0] = sns.histplot(x=dataframe[col_name], color="green", bins=10, ax=ax[0, 0])
        ax[0, 0].set_ylabel('Frequency')
        # ax[0, 0].set_title('Distribution')
        ax[0, 1] = plt.pie(dataframe[col_name].value_counts().values, labels=dataframe[col_name].value_counts().keys(),
                           colors=sns.color_palette('bright'), shadow=True, autopct='%.0f%%')
        plt.title("Percent")

        fig.set_size_inches(10, 6)
        fig.suptitle('Analysis of Categorical Variables', fontsize=13)
        plt.show()


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    # setup the plot grid
    if plot:
        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots(1, 2)
        ax = np.reshape(ax, (1, 2))
        ax[0, 0] = sns.histplot(x=dataframe[numerical_col], color="green", bins=20, ax=ax[0, 0])
        ax[0, 0].set_ylabel('Frequency')
        ax[0, 0].set_title('Distribution')
        ax[0, 1] = sns.boxplot(y=dataframe[numerical_col], color="purple", ax=ax[0, 1])
        ax[0, 1].set_title('Quantiles')

        fig.set_size_inches(10, 6)
        fig.suptitle('Analysis of Numerical Variables', fontsize=13)
        plt.show()


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def correlation_matrix(dataframe, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(dataframe[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


def correlated_map(dataframe, plot=False):
    corr = dataframe.corr()
    if plot:
        sns.set(rc={'figure.figsize': (12, 12)})
        sns.heatmap(corr, cmap="YlGnBu", annot=True, linewidths=.7)
        plt.xticks(rotation=60, size=15)
        plt.yticks(size=15)
        plt.title('Correlation Map', size=20)
        plt.show()

def target_correlation_matrix(dataframe, corr_th=0.5, target="Salary"):

    corr = dataframe.corr()
    corr_th = corr_th
    try:
        filter = np.abs(corr[target]) > corr_th
        corr_features = corr.columns[filter].tolist()
        sns.clustermap(dataframe[corr_features].corr(), annot=True, fmt=".2f")
        plt.show()
        return corr_features
    except:
        print("Hihg coraleted value, low the corr_th !")

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

# Importing Data and Check Data

df = pd.read_csv("datasets/hitters.csv")

# Check Data
check_df(df)


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (20, 20)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list



high_correlated_cols(df, plot=True)

df.groupby(["League", "Division"]).agg({"Salary":["mean", "median"],
                                        "Hits":["mean","median"],
                                        "Years":["mean", "median"],
                                        "CHits":["mean", "median"],
                                        "Assists":["mean", "median"],
                                        "Errors":["mean", "median"]})
# Analysis of Variables
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

# Analysis of Categorical Variables
for col in cat_cols:
    cat_summary(df, col, plot=True)

# Analysis of Numerical Variables
for col in num_cols:
    num_summary(df, col, plot=True)

# Analysis of Correlation
correlated_map(df, plot=True)
correlation_matrix(df, num_cols)
target_correlation_matrix(df, corr_th=0.5, target="Salary")

# Analysis of Categorical Variables with Target Variable
for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)



########################## 2. Data Preprocessing & Feature Engineering ##########################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

df.isnull().sum()
# Missing observations are only observed in the dependent variable, I delete them.
df.dropna(inplace=True)

df["Hits_Success"] = (df["AtBat"] - df["Hits"])
df["NEW_RBI_RATE"] = df["RBI"] / df["CRBI"]
df["NEW_HITS_RATE"] = df["Hits"] / df["CHits"]
df["NEW_WALKS_RATE"] = df["Walks"] / df["CWalks"]
df["NEW_WALKS_RATE"] = df["CHits"] / df["CAtBat"]
df["NEW_CRUNS_RATE"] = df["CRuns"] / df["Years"]
df["NEW_CHITS_RATE"] = df["CHits"] / df["Years"]

Putouts_label = ["littele_helper", "medium_helper", "very_helper"]
df["NEW_PUTOUTS_CAT"] = pd.qcut(df["PutOuts"], 3, labels=Putouts_label)

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)

############# Encoding  ##########################
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

len(binary_cols)

for col in binary_cols:
    label_encoder(df, col)


df = one_hot_encoder(df, cat_cols, drop_first=True)
check_df(df)


cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

################## Outlier Review  ##########################
for col in num_cols:
    print(col, check_outlier(df, col, 0.1, 0.9))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

################### Standartscaler  ##########################
num_cols.remove("Salary")

X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

y = df["Salary"]
X = df.drop(["Salary"], axis=1)

check_df(X)

df.isnull().sum()
df.shape
####################################### Main Procressing Function ##################################
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


df = pd.read_csv("datasets/hitters.csv")
X, y = hitters_data_pred(df)

X.head()
X.isnull().sum().sum()

############################################# RF MODELS AND FEATURE IMPORTANCE #############################################
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.20,
                                                   random_state=46)

rf_model = RandomForestRegressor(random_state=1).fit(X_train, y_train)

# Train Error
y_pred = rf_model.predict(X_train)
print("Train RMSE:", "{:,.4f}".format(np.sqrt(mean_squared_error(y_train, y_pred))), "\n")


# Test Error
y_pred2 = rf_model.predict(X_test)
print("Test RMSE:", "{:,.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred2))))


rf_model = RandomForestRegressor(random_state=17)

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 8, "auto"],
             "min_samples_split": [8, 13, 15, 20],
             "n_estimators": [100, 200, 250,]}


rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

rf_final = rf_model.set_params(**rf_best_grid.best_params_,
                               random_state=17).fit(X, y)

np.sqrt(-cross_val_score(rf_final, X, y, cv=10, scoring="neg_mean_squared_error"))
rmse = np.mean(np.sqrt(-cross_val_score(rf_final, X, y, cv=10, scoring="neg_mean_squared_error")))
print(f"RMSE: {round(rmse, 4)}")

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()

plot_importance(rf_final, X)


################################# Analyzing Model Complexity with Learning Curves (BONUS) ################################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.sqrt(np.mean(train_score, axis=1)**2)
    mean_test_score = np.sqrt(np.mean(test_score, axis=1)**2)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

rf_model = RandomForestRegressor(random_state=17)

rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]


for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1], "neg_root_mean_squared_error")


########################## 3. Base Models ############################
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

base_models(X, y)


#################### Automated Hyperparameter Optimization ###################################

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

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

best_models = {}

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

best_models = hyperparameter_optimization(X, y, 5)



######################## Stacking & Ensemble Learning #######################

def voting_regressorr(best_models, X, y):
    print("Voting Regressor")
    voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('LightGBM', best_models["LightGBM"])]).fit(X, y)

    cv_results = np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {cv_results}")

    return voting_reg

voting_reg = voting_regressorr(best_models, X, y)


######################## Prediction for a New Observation #######################

X.columns
random_user = X.sample(1, random_state=45)
voting_reg.predict(random_user)













