################################################
# KNN for Classification - ABC Grocery Task
################################################


################################################
# Import required packages
################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
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
# Feature Scaling
################################################

scale_norm = MinMaxScaler()
X_train = pd.DataFrame(scale_norm.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scale_norm.transform(X_test), columns = X_test.columns) 
""" Just transform for X_test no fit_transform
this means the scaling rules are being learned from train data and then apply to 
other data including our test sets and any other future data for predicting.
if we use fit_transform for X_test or any new datas, it will learn the rules base on that
and those rule maybe slightlydifferent each time, meaning we could not really rely on our predictions 
to be cosistent.
"""

################################################
# Feature Selection
################################################
from sklearn.ensemble import RandomForestClassifier

clf =RandomForestClassifier(random_state = 42)

feature_selector = RFECV(clf) 

fit = feature_selector.fit(X_train,y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features : {optimal_feature_count}") # Optimal number of features : 6

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

clf = KNeighborsClassifier()
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
accuracy_score(y_test, y_pred_class)  # 0.9363057324840764

# Precision (of all observations that were predicted as positive, how many were actually positive)
precision_score(y_test, y_pred_class) #  1.0

# Recall (of all positive obsevations, how many did we predict as positive )
recall_score(y_test, y_pred_class)    #   0.7619047619047619

# F1-score (the harmonic mean of precision and recall)
f1_score(y_test, y_pred_class)        #  0.8648648648648649


################################################
# Finding the optimal of k
################################################

k_list = list(range(2,25))
accuracy_scores = []


for k in k_list:
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred) 
    accuracy_scores.append(accuracy)
    

    
max_accuracy = max(accuracy_scores)  
max_accuracy_idx = accuracy_scores.index(max_accuracy) 
optimal_k_value = k_list[max_accuracy_idx] 


# Plot of max depths
plt.plot(k_list, accuracy_scores)
plt.scatter(optimal_k_value, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy (F1 score) by k \n Optimal Value for k: {optimal_k_value} (accuracy: {round(max_accuracy,4)})")
plt.xlabel("k")
plt.ylabel("Accuracy (F1_score)")
plt.tight_layout()
plt.show()
    

































