"""Code for migrating datasets
"""
import math

from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm


def migrate_dataset(
    old_token: str,
    new_token: str,
    dataset_id: str,
    new_dataset_id: Optional[str] = None,
    chunksize: int = 100,
    filters: list = None,
    show_progress_bar: bool = True,
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

    old_client = Client(token=old_token)
    new_client = Client(token=new_token)

    old_dataset = old_client.Dataset(dataset_id)
    new_dataset = new_client.Dataset(dataset_id)

    after_id = None

    number_of_documents = old_client.get_number_of_documents(
        dataset_id=dataset_id, filters=filters
    )
    iterations_required = math.ceil(number_of_documents / chunksize)
    bar = tqdm(range(iterations_required), disable=(not show_progress_bar))
    overflow: List[Dict[str, Any]] = []

    with FileLogger():
        while True:
            res = old_dataset.get_documents(
                number_of_documents=chunksize,
                filters=filters,
                after_id=after_id,
            )
            documents = res["documents"]
            after_id = res["after_id"]

            if not documents:
                break

            for document in documents:
                # backwards compatibility for clusters
                if "_clusters_" in document:
                    document["_cluster_"] = document.pop("_clusters_")

            pool = documents + overflow
            n_batches = int(len(pool) / chunksize)

            for i in range(n_batches):
                batch_start = i * chunksize
                batch_end = (i + 1) * chunksize

                batch = pool[batch_start:batch_end]
                res = new_dataset.insert_documents(batch)

                failed_documents = res["failed_documents"]

                failed_ids = set(map(lambda x: x["_id"], failed_documents))
                failed_documents = [
                    document for document in documents if document["_id"] in failed_ids
                ]
                overflow += failed_documents

            bar.update(1)

        if overflow:
            new_dataset.upsert_documents(overflow)
            bar.update(1)

    tqdm.write("Finished migrating.")
