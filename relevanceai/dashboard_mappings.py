"""Dashboard mappings
"""

# These are for where you need to submit an entire request and would require
# working with the frontend team to confirm these are working properly
DASHBOARD_MAPPINGS = {
    "multivector_search": "/sdk/search",
    "cluster_centroids_closest": "/sdk/cluster/centroids/closest",
    "cluster_centroids_furthest": "/sdk/cluster/centroids/furthest",
    "cluster_aggregation": "/sdk/cluster/aggregation",
}

# These are when you have a simple softlink
SOFTLINK_MAPPING = {"data_previwe": "/dataset/{dataset_id}/dashboard/data"}
