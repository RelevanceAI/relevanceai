#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#####
# Author: Charlene Leong charleneleong84@gmail.com
# Created Date: Monday, November 8th 2021, 8:15:18 pm
# Last Modified: Wednesday, November 10th 2021,1:53:47 am
#####
import pytest


@pytest.fixture(name='base_args')
def fixture_base_args():
    project = "dummy-collections"
    api_key = "UzdYRktIY0JxNmlvb1NpOFNsenU6VGdTU0s4UjhUR0NsaDdnQTVwUkpKZw"  # Read access
    base_url = "https://api-aueast.relevance.ai/v1/"
    base_args = {
                "project": project, 
                "api_key": api_key, 
                "base_url": base_url,
            }
    return base_args

@pytest.fixture(name='dataset_args')
def fixture_dataset_args():
    dataset_args = { 
        "dataset_id" : "ecommerce-6",
        "vector_field": "product_name_imagetext_vector_",
        "number_of_points_to_render" : 10,
        "random_state" : 0
    }
    return dataset_args


@pytest.fixture(name='test_args')
def fixture_test_args(base_args, dataset_args):
    test_args = {
    "dr" : 'pca',
    "dr_args" :  {
        "svd_solver": "auto",
        "random_state": 42
    },

    "vector_label" : "product_name",
    "vector_label_char_length"  : 12,

    "colour_label"  : None,  
    "colour_label_char_length" : 20,
    "hover_label": ["category"],
    
    "cluster": "kmeans",
    "cluster_args": {
        "init": "k-means++", 
        "verbose": 1,
        "compute_labels": True,
        "max_no_improvement": 2
    },
    "num_clusters": 10
    }
    return test_args
    

def test_plot(base_args, dataset_args, test_args):
    """Testing colour plot with cluster"""
    from relevanceai.http_client import Client

    client = Client(**base_args)
    client.projector.plot(**dataset_args, **test_args)

    assert True