"""Code for migrating datasets
"""
from typing import Optional


def migrate_dataset(
    old_token: str,
    new_token: str,
    dataset_id: str,
    new_dataset_id: Optional[str] = None,
    chunksize: int = 100,
    filters: list = None,
):
    """
    Migrate dataset

    Args
    ---------
        old_token (str): _description_
        new_token (str): _description_
        dataset_id (str): _description_
        new_dataset_id (Optional[str], optional): _description_. Defaults to None.
        chunksize (int, optional): _description_. Defaults to 20.

    Example
    ---------

    .. code-block::

        from relevanceai.migration import migrate
        migrate_dataset(
            old_token="...",
            new_token="...",
            dataset_id="sample_dataset",
            new_dataset_id="new_sample_dataset")

    """
    from relevanceai import Client
    from relevanceai.utils.logger import FileLogger

    filters = filters if filters is not None else []

    if new_dataset_id is None:
        new_dataset_id = dataset_id

    client = Client(token=old_token)
    ds = client.Dataset(dataset_id)
    with FileLogger():
        docs = ds.get_documents(
            number_of_documents=chunksize, include_cursor=True, filters=filters
        )
        while len(docs["documents"]) > 0:
            new_client = Client(token=new_token)
            new_ds = new_client.Dataset(new_dataset_id)
            for d in docs["documents"]:
                # backwards compatibility for clusters
                if "_clusters_" in d:
                    d["_cluster_"] = d.pop("_clusters_")
            new_ds.upsert_documents(docs["documents"])
            # we need to reset the config
            client = Client(token=old_token)
            docs = ds.get_documents(
                number_of_documents=chunksize,
                include_cursor=True,
                cursor=docs["cursor"],
                filters=filters,
            )
    print("Finished migrating.")
