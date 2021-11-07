"""API Client
"""
from loguru import logger as loguru_logger

from ..base import Base
from .admin import Admin
from .datasets import Datasets
from .services import Services


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class APIClient(Base):
    """API Client"""

    def __init__(self, project: str, api_key: str, base_url: str):
        self.datasets = Datasets(project=project, api_key=api_key, base_url=base_url)
        self.services = Services(project=project, api_key=api_key, base_url=base_url)
        self.admin = Admin(project=project, api_key=api_key, base_url=base_url)
        super().__init__(project, api_key, base_url)
