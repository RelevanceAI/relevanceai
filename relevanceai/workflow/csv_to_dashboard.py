"""
A simple CSV to dashboard workflow
"""
import pandas as pd
from typing import Optional, Dict, List
from relevanceai import ClusterOps


def df_to_dashboard(df: pd.DataFrame, text_cols: list):
    raise NotImplementedError


def text_csv_to_dashboard(
    csv_filename,
    text_cols: list,
    dataset_id: str,
    insertion_kwargs: Optional[dict] = None,
    text_encoder: Optional[dict] = None,
    vectorize_kwargs: Optional[dict] = None,
    cluster_model: Optional[str] = None,
    cluster_kwargs: Optional[dict] = None,
    token: Optional[str] = None,
):
    """
    This CSV takes in a file with a simple text column
    and provides a baseline insights/analytics dashboard.
    This can then be iterated on and improved.

    Parameters
    ------------
    csv_filename: str
        The CSV filename to insert
    text_columns: list
        The list of text columns
    """
    if cluster_kwargs is None:
        cluster_kwargs = {}
    if vectorize_kwargs is None:
        vectorize_kwargs = {}
    if insertion_kwargs is None:
        insertion_kwargs = {}

    from relevanceai import Client

    client = Client(token=token)

    print("Creating dataset...")
    ds = client.Dataset(dataset_id)

    print("Inserting CSV...")
    ds.insert_csv(csv_filename, **insertion_kwargs)

    print("Vectorize...")
    ds.vectorize(fields=text_cols, encoders={"text": text_encoder}, **vectorize_kwargs)

    print("Cluster...")
    cluster_ops: ClusterOps = ds.cluster(cluster_model, **cluster_kwargs)

    # Create suggested topics - summarize_closest alternative
    # Create a sample dashboard
    # Then summarize the closest ones
    print("Creating dashboard...")
    deployable_id = ds.deployables.create(dataset_id)
    print("Updating topics...")
    cluster_ops.summarize_closest(text_fields=text_cols, deployable_id=deployable_id)
