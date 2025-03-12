################################################
# Pipelines - Basic Template
################################################


# Import Required python packages

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#  Import sample data

my_df = pd.read_csv("data/pipeline_data.csv")

# split data into input and output objects

X = my_df.drop(["purchase"], axis = 1)
y = my_df["purchase"]

# split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42, stratify= y)



# Specify numeric and categorical feature

numeric_features = ["age", "credit_score"]
categorical_features = ["gender"]


################################################
# Set up Pipelines
################################################

# Numerical Feature Transform

numeric_transformer = Pipeline(steps = [("imputer", SimpleImputer()),
                                        ("scaler", StandardScaler())])


# Categorical Feature Transform

categorical_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy = "constant", fill_value = "U")),
                                        ("ohe", OneHotEncoder(handle_unknown = "ignore"))])

# Preprocessing Pipeline

preprocessing_pipeline = ColumnTransformer(transformers = [("numeric", numeric_transformer, numeric_features),
                                                           ("categorical", categorical_transformer, categorical_features)])


################################################
# Apply the Pipeline
################################################

# Logistic Regression

clf = Pipeline(steps = [("preprocessing_pipeline", preprocessing_pipeline),
                        ("classifier", LogisticRegression(random_state = 42))])
# X_train and others still have None valur and ... but pipelnie can handle it during fitting
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class) # 0.85

# Random Forest

clf = Pipeline(steps = [("preprocessing_pipeline", preprocessing_pipeline),
                        ("classifier", RandomForestClassifier(random_state = 42))])
# X_train and others still have None valur and ... but pipelnie can handle it during fitting
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class) # 0.85


################################################
# Save the pipeline
################################################

import joblib 
joblib.dump(clf, "data/model.joblib") # random forest model here


####################################################
# Import pipeline object and predict on new data
####################################################

# Import required pyrhon packages
import joblib
import pandas as pd
import numpy as np


# Import Pipeline

clf = joblib.load("data/model.joblib")


# Create new data

new_data = pd.DataFrame({"age": [25, np.nan, 50],
                         "gender" : ["M","F", np.nan],
                         "credit_score": [200, 100, 500]})


# pass new data in and receive predictions
clf.predict(new_data) # array([1, 0, 0], dtype=int64)



































