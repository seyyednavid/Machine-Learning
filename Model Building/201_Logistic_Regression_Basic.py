
################################################
# Logistic Regression - Basic Template
################################################

# Import required python package

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Import sample data

my_df = pd.read_csv("data/sample_data_classification.csv")


# split data into input and output objects

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]


# split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42, stratify = y)


# Instantiate our model object

clf = LogisticRegression(random_state = 42)


# Train our model

clf.fit(X_train, y_train)


# Asess model accuracy

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred) # 0.8

y_pred_prob = clf.predict_proba(X_test)
"""
array([[0.18627185, 0.81372815],
       [0.92633532, 0.07366468],
       [0.69737524, 0.30262476],
       [0.13673133, 0.86326867],
       [0.94374026, 0.05625974],
       [0.87816011, 0.1218398],
       ....)"""
       
conf_matrix = confusion_matrix(y_test, y_pred)      
print(conf_matrix) 
"""
[[8 3]
 [1 8]]
"""

plt.style.use("seaborn-v0_8-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center" , fontsize = 20)
plt.show()












