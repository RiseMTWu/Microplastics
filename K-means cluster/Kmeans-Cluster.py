# -*- coding: utf-8 -*-

#kmeans optimal k value determination method - elbow method

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('cluster-306.csv',encoding="GBK")

features =df.iloc[:,1:]

SSE = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k)  
    kmeans.fit(features)
    SSE.append(kmeans.inertia_) 
X = range(1,11)
plt.xlabel('number of clusters')
plt.ylabel('SSE')
plt.plot(X,SSE,'o-')
plt.show()
print(SSE)
#%%
# Cluster of different city groups
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('Cluster-306.csv',encoding="GBK")

features = df.iloc[:, 1:]

# K-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(features)

df['Cluster'] = kmeans.labels_

cluster_counts = df['Cluster'].value_counts().sort_index()
print(cluster_counts)

city_column = df.columns[0]

for cluster_id, group in df.groupby('Cluster', sort=True):
    print(f"\n第 {cluster_id} 组，共 {len(group)} 个城市：")
    print("、".join(group[city_column].astype(str)))
