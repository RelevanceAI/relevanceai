"""
Clusterer class to run clustering.
"""
from relevanceai.api.client import BatchAPIClient
from relevanceai.vector_tools.cluster import Cluster
from typing import Union, List
from relevanceai.dataset_api import Dataset

class Clusterer(BatchAPIClient):
    """Clusterer object designed to be a flexible class
    """
    def __init__(self, 
        model,
        dataset: Union[Dataset, str],
        vector_fields: List,
    ):
        self.dataset = dataset
        self.vector_fields = vector_fields
        self.model = model
    
    def fit(self):
        return self.dataset.fit_dataset()
