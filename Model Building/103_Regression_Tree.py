################################################
# Regression Tree - Basic Template
################################################


# Import required python package

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd



# Import sample data

my_df = pd.read_csv("data/sample_data_regression.csv")


# split data into input and output objects

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]


# split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42)


# Instantiate our model object

regressor = DecisionTreeRegressor(min_samples_leaf = 7)


# Train our model

regressor.fit(X_train, y_train)


# Asess model accuracy

y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred) #  0.5424730759242886


""" A Demonstration of Overfitting"""

y_pred_training = regressor.predict(X_train)
r2_score(y_train, y_pred_training) #  1.0 it has not generalized at all


""" Plot our Decision Tree"""

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = list(X.columns),
                 filled = True,
                 rounded = True,
                 fontsize = 24)



# one way for avoiding overfitting
# Instantiate our model object
regressor = DecisionTreeRegressor(min_samples_leaf = 7)

# Train our model
regressor.fit(X_train, y_train)

# Asess model accuracy
y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred) # 0.6570032913425767

y_pred_training = regressor.predict(X_train)
r2_score(y_train, y_pred_training)  #  0.7959618194306971

plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = list(X.columns),
                 filled = True,
                 rounded = True,
                 fontsize = 24)













