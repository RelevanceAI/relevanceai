from relevanceai import Client
from sklearn.cluster import KMeans

client = Client()

df = client.Dataset("iris")

n_clusters = 3
vector_field = "feature_vector_"
alias = "Species"

X = df.cluster(vector_field, 3, overwrite=True)
