# -*- coding: utf-8 -*-

from relevanceai.http_client import VecDBClient

# dataset_id = "ecommerce-6"
# project = "dummy-collections"
# api_key = "UzdYRktIY0JxNmlvb1NpOFNsenU6VGdTU0s4UjhUR0NsaDdnQTVwUkpKZw"  # Read access
# base_url = "https://api-aueast.relevance.ai/v1/"

# label = "product_name"
# vector_field = "product_name_imagetext_vector_"

# vi = VecDBClient(project, api_key, base_url=base_url)

# vi.services.projection.generate(
#             dataset_id=dataset_id, 
#             label=label, 
#             vector_field=vector_field,
#             dr='tsne'
#             )

import sys
sys.path.append('..')

from relevanceai.visualise.dataset import Dataset

dataset_id = "ecommerce-demo"
# project = "dummy-collections"
# api_key = "UzdYRktIY0JxNmlvb1NpOFNsenU6VGdTU0s4UjhUR0NsaDdnQTVwUkpKZw"  # Read access
project = '4219e219b6907fd6fbf0'
api_key = 'TWhLVU9Yd0JmQldJU2NLNXF5anM6TUNHcGhyd1FTV200RHdHbWRCaV9VUQ'

base_url = "https://api-aueast.relevance.ai/v1/"

data = Dataset(project, api_key, base_url=base_url, 
             dataset_id = dataset_id, number_of_documents = 1000)

data.df
