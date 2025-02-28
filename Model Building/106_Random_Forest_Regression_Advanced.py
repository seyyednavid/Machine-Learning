################################################
# Regression Tree - ABC Grocery Task
################################################


################################################
# Import required packages
################################################

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance



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



""" we don't need remove outliers when working with Random Forest Regression """




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


""" applying feature selection will not make any differenc to the performancee of a Random Forest in terms of
accuracy but it could help with performance in terms of computation"""


################################################
# Model Training
################################################

regressor = RandomForestRegressor(random_state = 42)
regressor.fit(X_train, y_train)

# Predict on the Test set

y_pred = regressor.predict(X_test)

# Calculate R-Squared

r_squared = r2_score(y_test, y_pred) #  0.9598627943571644


# Cross Validation
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2" )
cv_scores.mean()   #   0.9248589874052471

# Calculated Adjusted R-Squared
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1-r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1 )
# 0.9552756851408403



""" Feature Importance """

# first Method : Mean Decrease in Impurity (Gini Importance)

"""Measures how much each feature reduces uncertainty when making a decision.
Features that lead to more pure splits (more confident decisions) are more important.
"""
feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis = 1 )
feature_importance_summary.columns = ["input_variables", "feature_importance"]
feature_importance_summary.sort_values(by="feature_importance", inplace = True)

plt.barh(feature_importance_summary["input_variables"],feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()


# Other way for calculating Feature Importance is Permutation Importance
"""
Shuffles each feature and measures how much the model’s accuracy drops.
If accuracy drops significantly, the feature is highly important.
"""
result = permutation_importance(regressor, X_test, y_test, n_repeats= 10, random_state = 42)
# n_repeats =  Number of times to permute a feature

permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis = 1 )
permutation_importance_summary.columns = ["input_variables", "permutation_importance"]
permutation_importance_summary.sort_values(by="permutation_importance", inplace = True)

plt.barh(permutation_importance_summary["input_variables"],permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()


# How random forest work - Prediction under the hood

y_pred[0]  # 0.22624000000000016
new_data = [X_test.iloc[0]]
"""
[distance_from_store        2.270000
 credit_score               0.490000
 total_sales             1506.490000
 total_items              281.000000
 transaction_count         47.000000
 product_area_count         3.000000
 avarage_basket_value      32.052979
 gender_M                   0.000000
 Name: 0, dtype: float64]
"""
print(regressor.estimators_)
"""
[DecisionTreeRegressor(max_features=1.0, random_state=1608637542), 
 DecisionTreeRegressor(max_features=1.0, random_state=1273642419), 
 DecisionTreeRegressor(max_features=1.0, random_state=1935803228), 
 DecisionTreeRegressor(max_features=1.0, random_state=787846414), 
 DecisionTreeRegressor(max_features=1.0, random_state=996406378), 
 DecisionTreeRegressor(max_features=1.0, random_state=1201263687), 
...
"""

predictions = []
tree_count = 0

for tree in regressor.estimators_:
    prediction = tree.predict(new_data)[0]
    predictions.append(prediction)
    tree_count += 1


print(tree_count) # 100

print(predictions)    
"""
[0.201, 0.668, 0.117, 0.17, 0.203, 0.268, 0.17, 0.203, 0.23400000000000004, 0.17, 0.234, 
 0.12, 0.234, 0.174, 0.147, 0.352, 0.268, 0.234, 0.234, 0.234, 0.203, 0.17400000000000002, 
 0.17, 0.234, 0.177, 0.12, 0.314, 0.174, 0.234, 0.302, 0.352, 0.314, 0.234, 0.174, 0.116, 
 0.221, 0.147, 0.177, 0.203, 0.147, 0.12, 0.234, 0.234, 0.201, 0.177, 0.221, 0.203, 0.366, 
 0.201, 0.334, 0.366, 0.17, 0.448, 0.12, 0.234, 0.136, 0.23400000000000004, 0.17, 0.174, 
 0.201, 0.147, 0.201, 0.366, 0.344, 0.221, 0.17400000000000002, 0.234, 0.177, 0.17, 0.12, 
 0.42, 0.344, 0.12, 0.221, 0.17, 0.234, 0.17, 0.203, 0.234, 0.234, 0.314, 0.234, 0.234, 
 0.143, 0.17, 0.234, 0.201, 0.12, 0.174, 0.201, 0.35200000000000004, 0.234, 0.174, 0.506, 
 0.177, 0.352, 0.352, 0.147, 0.234, 0.201]
"""
sum(predictions) / tree_count # 0.22624000000000016


# Save Model
import pickle 

pickle.dump(regressor, open("data/random_forest_regression_model.p", "wb"))
pickle.dump(one_hot_encoder, open("data/random_forest_regression_ohe.p", "wb"))





























































