################################################
# Grid Search - Basic Template
################################################


# Import required python package

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import GridSearchCV


# Import sample data

my_df = pd.read_csv("data/sample_data_regression.csv")


# split data into input and output objects

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]


# Instantiate our GridSearch object

gscv = GridSearchCV(
    estimator = RandomForestRegressor(random_state = 42),
    param_grid = {"n_estimators" : [10, 50, 100, 500],
                  "max_depth" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]},
    cv = 5,
    scoring = "r2",
    # Number of jobs to run in parallel - None means 1 unless in a joblib.parallel_backend context. -1 means using all processors
    n_jobs = -1 
    )


# Fit to data

gscv.fit(X,y)



# Get the best CV score (mean)

gscv.best_score_ # 0.6472228691023957


# Optimal parameters

gscv.best_params_ # {'max_depth': 3, 'n_estimators': 500}



# Create optimal model object

regressor = gscv.best_estimator_ # RandomForestRegressor(max_depth=3, n_estimators=500, random_state=42)



















