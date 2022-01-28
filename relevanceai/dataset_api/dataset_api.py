"""
Pandas like dataset API
"""
import re
import math
import warnings
import pandas as pd
import numpy as np

from doc_utils import DocUtils

from typing import Dict, List, Union, Callable, Optional

from relevanceai.dataset_api.groupby import Groupby, Agg
from relevanceai.dataset_api.centroids import Centroids
from relevanceai.dataset_api.helpers import _build_filters

from relevanceai.vector_tools.client import VectorTools
from relevanceai.api.client import BatchAPIClient
from relevanceai.dataset_api.dataset_export import Export
from relevanceai.dataset_api.dataset_write import Write
from relevanceai.dataset_api.dataset_stats import Stats
from relevanceai.dataset_api.dataset_series import Series
from relevanceai.dataset_api.dataset_operations import Operations


class Dataset(Export, Stats, Operations):
    """Dataset class"""


class Datasets(BatchAPIClient):
    """Dataset class for multiple datasets"""

    def __init__(self, project: str, api_key: str):
        self.project = project
        self.api_key = api_key
        super().__init__(project=project, api_key=api_key)
