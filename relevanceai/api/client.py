"""API Client
"""
from relevanceai.base import Base
from relevanceai.api.admin import Admin
from relevanceai.api.datasets import Datasets
from relevanceai.api.services import Services

vis_requirements = False
try:
    from relevanceai.visualise.constants import *
    from relevanceai.visualise.dataset import Dataset
    from relevanceai.visualise.dim_reduction import DimReduction
    from relevanceai.visualise.cluster import Cluster
    from relevanceai.visualise.projector import Projector
    vis_requirements = True
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(e)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class APIClient(Base):
    """API Client"""

    def __init__(self, project: str, api_key: str, base_url: str):
        self.datasets = Datasets(project=project, api_key=api_key, base_url=base_url)
        self.services = Services(project=project, api_key=api_key, base_url=base_url)
        if vis_requirements:
            self.projector = Projector(project=project, api_key=api_key, base_url=base_url)
        self.admin = Admin(project=project, api_key=api_key, base_url=base_url)
        super().__init__(project, api_key, base_url)
