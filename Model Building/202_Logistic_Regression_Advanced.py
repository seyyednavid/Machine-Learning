################################################
# Logistic Regression - ABC Grocery Task
################################################


################################################
# Import required packages
################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV


################################################
# Import sample data
################################################

# Import

data_for_model = pd.read_pickle("data/abc_classification_modelling.p")

# Drop unnecessary columns

data_for_model.drop(["customer_id"], axis = 1, inplace = True)


# Shuffle data

data_for_model = shuffle(data_for_model, random_state = 42)


# Class Balance

data_for_model["signup_flag"].value_counts(normalize = True)
"""
signup_flag
0    0.689535
1    0.310465
Name: proportion, dtype: float64
"""

################################################
# Deal with missing values
################################################

data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)


################################################
# Deal with outliers
################################################

outlier_investigation = data_for_model.describe()

outlier_columns = ["distance_from_store", "total_sales", "total_items"]

# Boxplot approach

for column in outlier_columns:
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile -  lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    # Get index as I need to remove the whole row
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detectedin column {column}")
    
    # Remove the row of outliers from dataframe
    data_for_model.drop(outliers, inplace = True)

################################################
# Split Input Variables & Output Variables
################################################

X = data_for_model.drop(["signup_flag"], axis = 1)
y = data_for_model["signup_flag"]


################################################
# Split our Training & Test sets
################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)


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

################################################
# Feature Selection
################################################

clf =LogisticRegression(random_state = 42, max_iter = 1000)
""" max_iter has default 100.the number of iterations the model takes to try and find 
the optimal regression line """
feature_selector = RFECV(clf) # CV is default 5

fit = feature_selector.fit(X_train,y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features : {optimal_feature_count}") # Optimal number of features : 7

X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]

plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()


################################################
# Model Training
################################################

clf = LogisticRegression(random_state = 42, max_iter = 1000)
clf.fit(X_train, y_train)


################################################
# Model Assessment
################################################

y_pred_class = clf.predict(X_test)
# Probability of Class 1 (positive class)
y_pred_prob = clf.predict_proba(X_test)[:,1]


# Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred_class)      


plt.style.use("seaborn-v0_8-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center" , fontsize = 20)
plt.show()



# Accuracy (the number of correct classification out of all attempted classification)
accuracy_score(y_test, y_pred_class)  # 0.8662420382165605

# Precision (of all observations that were predicted as positive, how many were actually positive)
precision_score(y_test, y_pred_class) #  0.7837837837837838

# Recall (of all positive obsevations, how many did we predict as positive )
recall_score(y_test, y_pred_class)    #  0.6904761904761905

# F1-score (the harmonic mean of precision and recall)
f1_score(y_test, y_pred_class)        #  0.7341772151898734


################################################
# Finding the optimal threshold
################################################

thresholds = np.arange(0, 1, 0.01)

precision_scores = []
recall_scores = []
f1_scores = []

for threshold in thresholds:
    pred_class = (y_pred_prob >= threshold) * 1
    
    precision = precision_score(y_test, pred_class, zero_division=0)
    precision_scores.append(precision)
    
    recall = recall_score(y_test, pred_class)
    recall_scores.append(recall)
    
    f1 = f1_score(y_test, pred_class)
    f1_scores.append(f1)

max_f1 = max(f1_scores)  # 0.7804878048780488
max_f1_index = f1_scores.index(max_f1) #  44


plt.style.use("seaborn-v0_8-poster")
plt.plot(thresholds, precision_scores, label = "Precision", linestyle = "--")
plt.plot(thresholds, recall_scores, label = "Recall", linestyle = "--")
plt.plot(thresholds, f1_scores, label = "F1", linewidth = 5)
plt.title(f"Finding the Optimal Threshold foe Classification Model \n Max F1: {round(max_f1,2)} (Threshold = {round(thresholds[max_f1_index],2)})")
plt.xlabel("Threshold")
plt.ylabel("Assessment Score")
plt.legend(loc = "lower left")
plt.tight_layout()
plt.show()


optimal_threshold = 0.44
y_pred_class_opt_threshold = (y_pred_prob >= optimal_threshold) * 1
"""
array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
       1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
       0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 0])
"""





































