import joblib
import pandas as pd

df = pd.read_csv("datasets/hitters.csv")

random_user = df.sample(1, random_state=45)
new_model = joblib.load("C:/Users/Tuzcu/Desktop/DSMLBC/voting_clf_diabetes.pkl")

new_model.predict(random_user)


from week_08.homework.hitters_salary_prediction_pipeline import hitters_data_pred

X, y = hitters_data_pred(df)

random_user = X.sample(1, random_state=45)
new_model = joblib.load("/Users/Tuzcu/Desktop/DSMLBC/voting_clf_diabetes.pkl")
new_model.predict(random_user)
