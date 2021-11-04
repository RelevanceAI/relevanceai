# -*- coding: utf-8 -*-

from vecdb.http_client import VecDBClient

dataset_id = "ecommerce-6"
project = "dummy-collections"
api_key = "UzdYRktIY0JxNmlvb1NpOFNsenU6VGdTU0s4UjhUR0NsaDdnQTVwUkpKZw"  # Read access
base_url = "https://api-aueast.relevance.ai/v1/"

label = "product_name"
vector_field = "product_name_imagetext_vector_"

vi = VecDBClient(project, api_key, base_url=base_url)

vi.services.projection.generate(
            dataset_id=dataset_id, 
            label=label, 
            vector_field=vector_field,
            dr='tsne'
            )
