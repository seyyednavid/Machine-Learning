################################################
# Regression Tree - ABC Grocery Task
################################################


################################################
# Import required packages
################################################

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder



################################################
# Import sample data
################################################

# Import

data_for_model = pickle.load(open("data/abc_regression_modeling.p", "rb"))

# Drop unnecessary columns

data_for_model.drop(["customer_id"], axis = 1, inplace = True)


# Shuffle data

data_for_model = shuffle(data_for_model, random_state = 42)


################################################
# Deal with missing values
################################################

data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)



""" we don't need remove outliers when working with regression Tree """




################################################
# Split Input Variables & Output Variables
################################################

X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]


################################################
# Split our Training & Test sets
################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


################################################
# Deal with Categorical Variables
################################################

categorical_vars = ["gender"]

# If sparse_output=False, the output is a dense NumPy array
one_hot_encoder = OneHotEncoder(sparse_output= False, drop = "first")

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_features_name = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder_features_name)
X_train = pd.concat([X_train.reset_index(drop=True),X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(categorical_vars, axis = 1, inplace=True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder_features_name)
X_test = pd.concat([X_test.reset_index(drop=True),X_test_encoded.reset_index(drop=True)], axis=1)
X_test.drop(categorical_vars, axis = 1, inplace=True)


""" applying feature selection will not make any differenc to the performancee of a decision tree in terms of
accuracy but it could help with performance in terms of computation"""


################################################
# Model Training
################################################

regressor = DecisionTreeRegressor(random_state = 42)
regressor.fit(X_train, y_train)

# Predict on the Test set

y_pred = regressor.predict(X_test)

# Calculate R-Squared

r_squared = r2_score(y_test, y_pred) # 0.8981805706349476


# Cross Validation
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2" )
cv_scores.mean()   #  0.8760266035577694

# Calculated Adjusted R-Squared
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1-r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1 )
# 0.7535635576213854


""" A Demonstration of Overfitting"""

y_pred_training = regressor.predict(X_train)
r2_score(y_train, y_pred_training) #  1.0 it has not generalized at all




# Finding the best max_depth

max_depth_list = list(range(1,9))
accuracy_scores = []


for depth in max_depth_list:
    regressor = DecisionTreeRegressor(random_state = 42, max_depth = depth)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy = r2_score(y_test, y_pred) 
    accuracy_scores.append(accuracy)
    
""" [0.48041516747913704,
 0.749628862008641,
 0.8434993278383679,
 0.8666832224200037,
 0.8909700995376669,
 0.8905579006885826,
 0.8990238112614182,
 0.8941316438857448] """
    
max_accuracy = max(accuracy_scores) # 0.8990238112614182
max_accuracy_idx = accuracy_scores.index(max_accuracy) # 6
optimal_depth = max_depth_list[max_accuracy_idx] # 7


# Plot of max depths
plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy by Max Depth \n Optimal Tree Depth: {optimal_depth} (accuracy: {round(max_accuracy,4)})")
plt.xlabel("Max depth of Decision Tree")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()
    

# Plot our model

plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = list(X.columns),
                 filled = True,
                 rounded = True,
                 fontsize = 16)















































