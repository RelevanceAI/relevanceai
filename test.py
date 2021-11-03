# -*- coding: utf-8 -*-

from vecdb.vis.projection import Projection

dataset_id = "ecommerce-6"
project = "dummy-collections"
api_key = "UzdYRktIY0JxNmlvb1NpOFNsenU6VGdTU0s4UjhUR0NsaDdnQTVwUkpKZw"  # Read access
base_url = "https://ingest-api-dev-aueast.relevance.ai/latest/"

Projection(dataset_id=dataset_id, project=project, api_key=api_key, base_url=base_url)
