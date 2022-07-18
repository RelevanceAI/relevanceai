from sklearn.cluster import KMeans
from relevanceai.apps.recipes.model_observability.cluster.evaluation import ClusterEvaluation

def elbow_method(
    X, 
    clusters_to_try=range(1, 10), 
    full_evaluation:bool=False, 
    **kwargs
):
    for k in clusters_to_try:
        model = KMeans(n_clusters=k, **kwargs)
        cluster_labels = model.fit(X)
        ClusterEvaluation(X, cluster_labels)
    return 

def dendrogram_method(X):
    return 