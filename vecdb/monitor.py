"""All Dataset related functions
"""
from .base import Base

class Monitor(Base):
    def __init__(self, project, api_key, base_url):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
