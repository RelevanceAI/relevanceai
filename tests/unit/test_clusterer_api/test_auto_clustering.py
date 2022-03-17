import time
import pytest
from relevanceai import Client
from relevanceai.interfaces import ClusterOps
from ...utils import is_subname_in_schema


def test_minibatch_clusterer(minibatch_clusterer: ClusterOps):
    assert len(minibatch_clusterer.list_closest_to_center()) > 0
    assert len(minibatch_clusterer.centroids) > 0


def test_minibatchkmeans_cluster_report(minibatch_clusterer: ClusterOps):
    assert len(minibatch_clusterer.internal_report()) > 0


def test_kmeans_clusterer(kmeans_clusterer: ClusterOps):
    assert len(kmeans_clusterer.list_closest_to_center()) > 0
    assert len(kmeans_clusterer.centroids) > 0


@pytest.mark.skip(reason="tofix for jacky wong")
def test_minibatchkmeans_subclusterer(minibatch_subclusterer):
    time.sleep(2)
    df, ALIAS = minibatch_subclusterer
    assert is_subname_in_schema(ALIAS, list(df.schema))


@pytest.mark.skip(reason="tofix for jacky wong")
def test_kmeans_subclusterer(kmeans_subclusterer):
    time.sleep(2)
    df, ALIAS = kmeans_subclusterer
    assert is_subname_in_schema(ALIAS, list(df.schema))


def test_kmeans_cluster_report(kmeans_clusterer: ClusterOps):
    assert len(kmeans_clusterer.internal_report()) > 0


def test_storing_reports(test_client: Client, kmeans_clusterer: ClusterOps):
    report = kmeans_clusterer.internal_report()
    result = test_client.store_cluster_report(report_name="simple", report=report)
    time.sleep(2)
    assert len(test_client.list_cluster_reports()) > 1
    # Now delete the rep0ort - provide some cleanup :)
    test_client.delete_cluster_report(result["_id"])
