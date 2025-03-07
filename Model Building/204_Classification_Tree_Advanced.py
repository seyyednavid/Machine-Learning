################################################
# Classification Tree - ABC Grocery Task
################################################


################################################
# Import required packages
################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder



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
# Model Training
################################################

clf = DecisionTreeClassifier(random_state = 42, max_depth = 5)
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
accuracy_score(y_test, y_pred_class)  # 0.9294117647058824

# Precision (of all observations that were predicted as positive, how many were actually positive)
precision_score(y_test, y_pred_class) #   0.8846153846153846

# Recall (of all positive obsevations, how many did we predict as positive )
recall_score(y_test, y_pred_class)    #  0.8846153846153846

# F1-score (the harmonic mean of precision and recall)
f1_score(y_test, y_pred_class)        #   0.8846153846153846



# Finding the best max_depth

max_depth_list = list(range(1,15))
accuracy_scores = []


for depth in max_depth_list:
    clf = DecisionTreeClassifier(random_state = 42, max_depth = depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred) 
    accuracy_scores.append(accuracy)
    

    
max_accuracy = max(accuracy_scores)  # 0.9245283018867925
max_accuracy_idx = accuracy_scores.index(max_accuracy) # 8 
optimal_depth = max_depth_list[max_accuracy_idx] 


# Plot of max depths
plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy (F1 score) by Max Depth \n Optimal Tree Depth: {optimal_depth} (accuracy: {round(max_accuracy,4)})")
plt.xlabel("Max depth of Decision Tree")
plt.ylabel("Accuracy (F1_score)")
plt.tight_layout()
plt.show()
    

# Plot our model

plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = list(X.columns),
                 filled = True,
                 rounded = True,
                 fontsize = 16)









































