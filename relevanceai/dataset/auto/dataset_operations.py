# -*- coding: utf-8 -*-
"""
Pandas like dataset API
"""
from relevanceai.dataset.auto.cluster import Cluster
from relevanceai.dataset.auto.labels import Labels
from relevanceai.dataset.auto.reduce_dimensions import DimensionalityReduction
from relevanceai.dataset.auto.vectorize import Vectorize
from relevanceai.dataset.auto.community_detection import CommunityDetection
from relevanceai.dataset.auto.scaling import Scaler


class Operations(
    Cluster, Labels, DimensionalityReduction, Vectorize, CommunityDetection, Scaler
):
    pass
