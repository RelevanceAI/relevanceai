#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#####
# Author: Charlene Leong charleneleong84@gmail.com
# Created Date: Monday, November 8th 2021, 8:15:18 pm
# Last Modified: Thursday, November 11th 2021,3:54:35 am
#####
import pytest
from parameterized import parameterized

DATASET_ARGS =[ 
    { 
    "dataset_id" : "ecommerce-6",
    "vector_field": "product_name_imagetext_vector_",
    "number_of_points_to_render" : 10,
    "random_state" : 0
    },
    { 
    "dataset_id" : "unsplash-images",
    "vector_field": "image_url_vector_",
    "number_of_points_to_render" : 10,
    "random_state" : 42
    },
]
@pytest.fixture(name='base_args',
params=[{
    "project": "dummy-collections", 
    "api_key": "UzdYRktIY0JxNmlvb1NpOFNsenU6VGdTU0s4UjhUR0NsaDdnQTVwUkpKZw",  # Read access, 
    "base_url": "https://api-aueast.relevance.ai/v1/",
}
])
def fixture_base_args(request):
    return request.param

@pytest.fixture(name='dataset_args', 
params =[ 
        { 
        "dataset_id" : "ecommerce-6",
        "vector_field": "product_name_imagetext_vector_",
        "number_of_points_to_render" : 100,
        "random_state" : 0,
        "vector_label" : "product_name",
        "hover_label": ["category"],
        },
        { 
        "dataset_id" : "unsplash-images",
        "vector_field": "image_url_vector_",
        "number_of_points_to_render" : 100,
        "random_state" : 42
        },
    ])
def fixture_dataset_args(request):
    return request.param


@pytest.fixture(name='test_args',
params=[{
    "dr" : 'pca',
    "dr_args" :  {
        "svd_solver": "auto",
        "random_state": 42
    },
    "vector_label_char_length"  : 12,

    "colour_label"  : None,
    "colour_label_char_length" : 20,
    "cluster": "kmeans",
    "cluster_args": {
        "init": "k-means++", 
        "verbose": 1,
        "compute_labels": True,
        "max_no_improvement": 2
    },
    "num_clusters": 10
    }
])
def fixture_test_args(request):
    return request.param
    


def test_plot(base_args, dataset_args, test_args):
    """Testing colour plot with cluster"""
    from relevanceai.http_client import Client
    
    client = Client(**base_args)
    client.projector.plot(**dataset_args, **test_args)

    import json
    from pprint import pprint
    pprint(json.dumps({**dataset_args, **test_args}, indent=4))

    assert True