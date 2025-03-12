################################################
# K Means Clustering - Basic Template
################################################


# Import required Python packages

from sklearn.cluster import KMeans
import pandas as pd 
import matplotlib.pyplot as plt

# Import sample data

my_df = pd.read_csv("data/sample_data_clustering.csv")


# Plot the data

plt.scatter(my_df["var1"], my_df["var2"])
plt.xlabel("var1")
plt.ylabel("var2")
plt.show()


# Instantiate & fit the model

kmeans = KMeans(n_clusters = 3, random_state = 42)
kmeans.fit(my_df)


# Add the cluster Labels to our df

my_df["cluster"] = kmeans.labels_
my_df["cluster"].value_counts()
"""
cluster
0    100
2    100
1    100
Name: count, dtype: int64
"""

# Plot our clusters and centroids
centroids = kmeans.cluster_centers_
print(centroids)
"""
[[0.61145409 0.82340359]
 [0.20990899 0.58086933]
 [0.76730971 0.25649517]]
"""


clusters = my_df.groupby("cluster")

for cluster, data in clusters:
    plt.scatter(data["var1"], data["var2"], marker = "o", label = cluster)
    plt.scatter(centroids[cluster, 0], centroids[cluster, 1], marker="x", color="black", s = 300)
plt.legend()
plt.tight_layout()
plt.show()