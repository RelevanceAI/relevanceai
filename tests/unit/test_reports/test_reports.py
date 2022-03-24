import requests
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from relevanceai.reports.cluster.report import ClusterReport


def test_cluster_reporting_smoke():
    docs = requests.get(
        "https://raw.githubusercontent.com/fanzeyi/pokemon.json/master/pokedex.json"
    ).json()
    for d in docs:
        b = d["base"]
        d.update(b)
        d["base_vector_"] = [
            b["Attack"],
            b["Defense"],
            b["HP"],
            b["Sp. Attack"],
            b["Sp. Defense"],
            b["Speed"],
        ]

    df = pd.DataFrame(docs)
    vectors = np.array(df["base_vector_"].tolist())

    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(vectors)

    report = ClusterReport(
        X=vectors, cluster_labels=cluster_labels, num_clusters=n_clusters, model=kmeans
    )

    internal_report = report.internal_report
    assert len(internal_report) >= 0
