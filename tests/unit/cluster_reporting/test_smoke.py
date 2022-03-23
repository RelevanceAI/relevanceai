import requests


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

    import pandas as pd
    import numpy as np

    df = pd.DataFrame(docs)
    X = np.array(df["base_vector_"].tolist())

    from relevanceai.core.cluster.reports.cluster_report.cluster_report import (
        ClusterReport,
    )
    from sklearn.cluster import KMeans

    N_CLUSTERS = 2
    kmeans = KMeans(n_clusters=N_CLUSTERS)
    cluster_labels = kmeans.fit_predict(X)

    report = ClusterReport(
        X=X, cluster_labels=cluster_labels, num_clusters=N_CLUSTERS, model=kmeans
    )

    internal_report = report.internal_report
    assert len(internal_report) >= 0
