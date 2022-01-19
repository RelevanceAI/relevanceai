from relevanceai import Client
from sklearn.cluster import KMeans

client = Client()

df = client.Dataset('palmer-penguins')
n_clusters = 10
vector_field = 'Culmen Depth (mm)'
alias = 'Culmen Depth'

X = df[vector_field].numpy()
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
df.centroids[vector_field][alias] = kmeans.cluster_centers_

df.centroids.insert(vector_field, alias) = kmeans.cluster_centers_
df.centroids.insert.metadata(vector_field, alias, n_clusters, etc) = kmeans.labels_