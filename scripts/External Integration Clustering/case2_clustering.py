from relevanceai.vector_tools.cluster import Clusterbase
from sklearn.cluster import KMeans
 
class CLusterer(ClusterBase):
  def __init__(self, clusterer):
     self.clusterer = clusterer
  
  # Require users to build this
  def fit_transform(self, X):
     return self.clusterer.fit_transform(X)
  
  def get_centroids(self, x):
     return self.clusterer.cluster_centers_
  
  def metadata(self):
     return {"n_clusters": 2, "random_state": 0}

# Relevance AI handles the metadata,
kmeans = KMeans(n_clusters=2, random_state=0)
clusterer = CLusterer(kmeans)
clusterer.fit_dataset(df)