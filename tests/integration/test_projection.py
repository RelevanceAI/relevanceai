"""Pytests for the projector - not really good
"""
import pytest
from typing_extensions import get_args

from relevanceai.vector_tools.constants import DIM_REDUCTION, DIM_REDUCTION_DEFAULT_ARGS
from relevanceai.vector_tools.constants import CLUSTER, CLUSTER_DEFAULT_ARGS


@pytest.fixture(
    name="dataset_args",
    params=[
        {  ## Testing vector label and empty hover label
            "vector_field": "sample_1_vector_",
            "vector_label": "sample_1_label",
            "vector_label_char_length": 12,
            "number_of_points_to_render": 100,
            "random_state": 0,
        },
        {  ## Testing colour label
            "vector_field": "sample_2_vector_",
            "colour_label": "sample_2_label",
            "colour_label_char_length": 20,
            "number_of_points_to_render": 100,
            "random_state": 42,
        },
        {  ## Testing vector label, colour label and hover label
            "vector_field": "sample_3_vector_",
            "colour_label": "sample_3_label",
            "colour_label_char_length": 20,
            "hover_label": ["sample_1_label", "sample_2_label", "sample_3_label"],
            "number_of_points_to_render": None,
            "random_state": 42,
        },
    ],
)
def fixture_dataset_args(vector_dataset_id, request):
    return {"dataset_id": vector_dataset_id, **request.param}


@pytest.fixture(
    name="dr_args",
    params=[
        {"dr": dr, "dr_args": {**DIM_REDUCTION_DEFAULT_ARGS[dr]}}
        for dr in get_args(DIM_REDUCTION)
    ],
)
def fixture_dr_args(request):
    return request.param


@pytest.fixture(
    name="cluster_args",
    params=[
        {"cluster": c, "cluster_args": {**CLUSTER_DEFAULT_ARGS[c]}}
        for c in get_args(CLUSTER)
        if c
    ],
)
def fixture_cluster_args(request):
    return request.param


# @pytest.mark.skip(reason = 'Slow')
# def test_projector_plot(test_client: Client, dataset_args, dr_args, cluster_args):
#     """Testing vector label with cluster"""
#     test_client.projector.plot(**dataset_args, **dr_args, **cluster_args)
#     assert True

# def test_projector_plot_fast(test_df: Dataset):
#     test_client.projector.plot(vector_dataset_id, "sample_1_vector_", colour_label = "sample_1_label", cluster = 'kmeans', dims = 2, number_of_points_to_render = 100)
#     assert True
