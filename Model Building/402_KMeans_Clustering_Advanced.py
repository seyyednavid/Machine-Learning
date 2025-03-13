################################################
# K Means Clustering - Advanced Template
################################################


# Import required Python packages

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

################################################
# Create the data
################################################

# import tables

transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name = "transactions")
product_areas = pd.read_excel("data/grocery_database.xlsx", sheet_name = "product_areas")

# merge om product area name

transactions = pd.merge(transactions, product_areas, how = "inner", on = "product_area_id" )

# drop the non-food category

transactions.drop(transactions[transactions["product_area_name"] == "Non-Food"].index, inplace = True)

# aggregate sales at customer Level (by product area)

transactions_summary = transactions.groupby(["customer_id", "product_area_name"])["sales_cost"].sum().reset_index()

# pivot data to place product areas as column

transactions_summary_pivot = transactions.pivot_table(index = "customer_id",
                                                      columns = "product_area_name",
                                                      values = "sales_cost",
                                                      aggfunc = "sum",
                                                      fill_value = 0,
                                                      margins = True,
                                                      margins_name = "Total").rename_axis(None, axis = 1)


# Turn sales into % sales 

transactions_summary_pivot = transactions_summary_pivot.div(transactions_summary_pivot["Total"], axis = 0)

# drop the total column

data_for_clustering = transactions_summary_pivot.drop(["Total"], axis = 1)

data_for_clustering = data_for_clustering.drop("Total", axis=0) # remove total in last row


################################################
# Data preperation & cleaning 
################################################


# check for missing values

data_for_clustering.isna().sum()

# normalize data

scale_norm = MinMaxScaler()
data_for_clustering_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustering), columns=data_for_clustering.columns)


################################################
# Use Wcss to find a good value for k
################################################

k_values = list(range(1, 10))
wcss_list = []

for k in k_values:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(data_for_clustering_scaled)
    wcss_list.append(kmeans.inertia_)

plt.plot(k_values, wcss_list)
plt.title("Within cluster sum of squares - by k")
plt.xlabel("k")
plt.ylabel("WCSS score")
plt.tight_layout()
plt.show() # k = 3


################################################
# Instantiate & fit model
################################################

kmeans = KMeans(n_clusters = 3, random_state = 42, n_init = 10)
kmeans.fit(data_for_clustering_scaled)


################################################
# Use cluster information
################################################

# Add cluster labels to our data

data_for_clustering["cluster"] = kmeans.labels_


# check cluster sizee

data_for_clustering["cluster"].value_counts()
"""
cluster
1    636
2    132
0    102
Name: count, dtype: int64
"""

################################################
# Profile our clusters
################################################

cluster_summary = data_for_clustering.groupby("cluster")[["Dairy", "Fruit", "Meat", "Vegetables"]].mean().reset_index()




















































 

























































