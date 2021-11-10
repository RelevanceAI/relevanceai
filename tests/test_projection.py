#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#####
# Author: Charlene Leong charleneleong84@gmail.com
# Created Date: Monday, November 8th 2021, 8:15:18 pm
# Last Modified: Wednesday, November 10th 2021,1:23:42 am
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
        "number_of_documents" : 10,
        "random_state" : 0
    }
    return dataset_args


@pytest.fixture(name='test_args')
def fixture_test_args(base_args, dataset_args):
    test_args = {
    "number_of_points_to_render":100,
    "dr" : 'pca',
    "dr_args" :  {
        "svd_solver": "auto",
        "random_state": 42
    },
    "vector_label" : "product_name",
    "vector_label_char_length"  : 12,
    "vector_field" : 'product_name_imagetext_vector_',
    "colour_label"  : None,  
    "colour_label_char_length" : 20,
    "hover_label": ["category"],
    "cluster": "kmeans",
    "cluster_args": {
        "init": "Huang", 
        "verbose": 1,
        "random_state": 42,
        "n_jobs": -1
    },
    "num_clusters": 10
    }
    return test_args
    


# def test_retrieve_datasets(base_args, dataset_args, test_args):
#     from relevanceai.visualise.dataset import Dataset
    
#     dataset = Dataset(**base_args, **dataset_args)
    
    # assert dataset.dataset_id == dataset_args["dataset_id"]
    # assert dataset.number_of_documents == dataset_args["number_of_documents"]
    # assert dataset.random_state == dataset_args["random_state"]

    # assert dataset.vector_fields == dataset_args["vector_fields"]

    # dataset = Dataset(base_args, 
    #                     dataset_id=dataset_id, number_of_documents=number_of_documents, 
    #                     random_state=random_state
    #                     )



def test_plot(base_args, dataset_args, test_args):
    from relevanceai.http_client import Client

    client = Client(**base_args)
    client.projector.plot(**dataset_args, **test_args)

    assert True