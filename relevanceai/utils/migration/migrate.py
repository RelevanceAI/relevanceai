"""Code for migrating datasets
"""
import math

from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm

from relevanceai.operations_new.ops_run import PullTransformPush


def migrate_dataset(
    old_token: str,
    new_token: str,
    dataset_id: str,
    new_dataset_id: Optional[str] = None,
    chunksize: Optional[int] = None,
    filters: list = None,
    show_progress_bar: bool = True,
    max_workers: int = 1,
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

    filters = filters if filters is not None else []

    if new_dataset_id is None:
        new_dataset_id = dataset_id

    old_client = Client(token=old_token)
    new_client = Client(token=new_token)

    old_dataset = old_client.Dataset(dataset_id)
    new_dataset = new_client.Dataset(dataset_id)

    ptp = PullTransformPush(
        pull_dataset=old_dataset,
        push_dataset=new_dataset,
        func=None,
        show_progress_bar=show_progress_bar,
        show_transform_progress_bar=False,
        pull_chunksize=chunksize,
        push_chunksize=chunksize,
        push_workers=max_workers,
    )
    ptp.run()

    metadata = old_dataset.metadata.to_dict()
    new_dataset.upsert_metadata(metadata)

    cluster_fields = [
        field
        for field in old_dataset.schema
        if "_cluster_" in field and len(field.split(".")) == 3
    ]
    if cluster_fields:
        tqdm.write(f"Found {len(cluster_fields)} cluster_fields")

    for cluster_field in cluster_fields:
        _, vector_field, alias = cluster_field.split(".")

        tqdm.write(f"Inserting Centroids for `{cluster_field}`...")
        old_cluster_ops = old_client.ClusterOps(
            alias=alias,
            vector_fields=[vector_field],
            dataset_id=old_dataset.dataset_id,
        )
        centroid_documents = old_cluster_ops.centroids
        new_dataset.datasets.cluster.centroids.insert(
            dataset_id=new_dataset.dataset_id,
            cluster_centers=new_dataset.json_encoder(centroid_documents),
            vector_fields=[vector_field],
            alias=alias,
        )

    tqdm.write("Centroids inserted!")

    tqdm.write("Migrating Operations Metadata...")
    metadata = old_dataset.metadata.to_dict()
    new_dataset.insert_metadata(metadata)
    tqdm.write("Operations Metadata Migrated!")

    tqdm.write("Finished migrating.")
