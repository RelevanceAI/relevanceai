"""Code for migrating datasets
"""
from typing import Optional

def migrate_dataset(old_token: str, new_token: str, dataset_id: str, new_dataset_id: Optional[str] = None,
    chunksize: int=20)
    from relevanceai import Client
    from relevanceai.package_utils.logger import FileLogger
    if new_dataset_id is None: new_dataset_id = dataset_id

    client = Client(token=old_token)
    ds = client.Dataset(dataset_id)
    with FileLogger():
        docs = ds.get_documents(number_of_documents=chunksize,include_cursor=True)
        while len(docs['documents']) > 0:
            new_client = Client(token=new_token)
            new_ds  = new_client.Dataset(new_dataset_id)
            for d in docs['documents']:
                # backwards compatibility for clusters
                if "_clusters_" in d:
                    d['_cluster_'] = d.pop("_clusters_")
            new_ds.upsert_documents(docs['documents'])
            # we need to reset the config
            client = Client(token=old_token)
            docs = ds.get_documents(
                number_of_documents=chunksize,
                include_cursor=True, cursor=docs['cursor'])
    print("Finished migrating.")
