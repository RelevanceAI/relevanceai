"""API Client
"""

from vecdb.api.admin import Admin
from vecdb.api.datasets import Datasets
from vecdb.api.services import Services
from vecdb.base import Base

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class APIClient(Base):
    """API Client"""

    def __init__(self, project: str, api_key: str, base_url: str):
        self.datasets = Datasets(project=project, api_key=api_key, base_url=base_url)
        self.services = Services(project=project, api_key=api_key, base_url=base_url)
        self.admin = Admin(project=project, api_key=api_key, base_url=base_url)
        super().__init__(project, api_key, base_url)
