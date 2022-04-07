.. code:: ipython3

    from relevanceai import Client

.. code:: ipython3

    client = Client()


.. parsed-literal::

    Activation token (you can find it here: https://cloud.relevance.ai/sdk/api )

    Connecting to us-east-1...
    You can view all your datasets at https://cloud.relevance.ai/datasets/
    Welcome to RelevanceAI. Logged in as 59066979f4876d91beea.


.. code:: ipython3

    from relevanceai.utils.datasets import get_iris_dataset

    dataset = client.Dataset("iris")

    documents = get_iris_dataset()

    dataset.insert_documents(documents, create_id=True)


.. parsed-literal::

    while inserting, you can visit your dashboard at https://cloud.relevance.ai/dataset/iris/dashboard/monitor/
    ✅ All documents inserted/edited successfully.


.. code:: ipython3

    dataset.schema




.. parsed-literal::

    {'PetalLengthCm': 'numeric',
     'PetalWidthCm': 'numeric',
     'SepalLengthCm': 'numeric',
     'SepalWidthCm': 'numeric',
     'Species': 'text',
     '_cluster_': 'dict',
     '_cluster_.feature_vector_': 'dict',
     '_cluster_.feature_vector_.kmeans-3': 'text',
     '_cluster_.feature_vector_.kmeans-6': 'text',
     'feature_vector_': {'vector': 4},
     'insert_date_': 'date'}



.. code:: ipython3

    dataset.cat("feature_vector_", fields=list(dataset.schema)[:4])


.. parsed-literal::

    📌 Your logs have been saved to iris_06-04-2022-14-08-30_pull_update_push.log. If you are debugging, you can turn file logging off by setting `log_to_file=False`.📌


.. code:: ipython3

    dataset.schema




.. parsed-literal::

    {'PetalLengthCm': 'numeric',
     'PetalWidthCm': 'numeric',
     'SepalLengthCm': 'numeric',
     'SepalWidthCm': 'numeric',
     'Species': 'text',
     '_cluster_': 'dict',
     '_cluster_.feature_vector_': 'dict',
     '_cluster_.feature_vector_.kmeans-3': 'text',
     '_cluster_.feature_vector_.kmeans-6': 'text',
     'feature_vector_': {'vector': 4},
     'insert_date_': 'date'}



.. code:: ipython3

    dataset.cluster(model="kmeans", n_clusters=6, vector_fields=["feature_vector_"])


.. parsed-literal::

    100%|██████████| 1/1 [00:00<00:00,  2.62it/s]
    100%|██████████| 1/1 [00:01<00:00,  1.78s/it]


.. parsed-literal::

    Build your clustering app here: https://cloud.relevance.ai/dataset/iris/deploy/recent/cluster/


.. code:: ipython3

    from relevanceai import ClusterOps

    ops = ClusterOps.from_dataset(dataset, alias="kmeans-6", vector_fields=["feature_vector_"])

    ops.merge(cluster_labels=(1, 3), alias="kmean-6")


.. parsed-literal::

    Response failed (https://api.us-east-1.relevance.ai/latest/datasets/iris/cluster/centroids/documents) (Status: 404 Response: {"message":"No centroids were found for chosen vector fields and alias."})


::


    ---------------------------------------------------------------------------

    APIError                                  Traceback (most recent call last)

    c:\Users\Joseph\Documents\Jobs\RELEVANCE AI\RelevanceAI\guides\cluster_merging_guide.ipynb Cell 8' in <module>
          <a href='vscode-notebook-cell:/c%3A/Users/Joseph/Documents/Jobs/RELEVANCE%20AI/RelevanceAI/guides/cluster_merging_guide.ipynb#ch0000008?line=0'>1</a> from relevanceai import ClusterOps
          <a href='vscode-notebook-cell:/c%3A/Users/Joseph/Documents/Jobs/RELEVANCE%20AI/RelevanceAI/guides/cluster_merging_guide.ipynb#ch0000008?line=2'>3</a> ops = ClusterOps.from_dataset(dataset, alias="kmeans-6", vector_fields=["feature_vector_"])
    ----> <a href='vscode-notebook-cell:/c%3A/Users/Joseph/Documents/Jobs/RELEVANCE%20AI/RelevanceAI/guides/cluster_merging_guide.ipynb#ch0000008?line=4'>5</a> ops.merge(cluster_labels=(1, 3), alias="kmean-6")


    File c:\users\joseph\documents\jobs\relevance ai\relevanceai\relevanceai\utils\decorators\analytics.py:116, in track.<locals>.wrapper(*args, **kwargs)
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/decorators/analytics.py?line=113'>114</a>     pass
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/decorators/analytics.py?line=114'>115</a> try:
    --> <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/decorators/analytics.py?line=115'>116</a>     return func(*args, **kwargs)
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/decorators/analytics.py?line=116'>117</a> finally:
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/decorators/analytics.py?line=117'>118</a>     os.environ[TRANSIT_ENV_VAR] = "FALSE"


    File c:\users\joseph\documents\jobs\relevance ai\relevanceai\relevanceai\operations\cluster\cluster.py:768, in ClusterOps.merge(self, cluster_labels, alias, show_progress_bar)
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/operations/cluster/cluster.py?line=764'>765</a>     alias = "communitydetection"
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/operations/cluster/cluster.py?line=765'>766</a>     print("No alias given, assuming `communitydetection`")
    --> <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/operations/cluster/cluster.py?line=767'>768</a> centroid_documents = self.services.cluster.centroids.list(
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/operations/cluster/cluster.py?line=768'>769</a>     dataset_id=self.dataset_id,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/operations/cluster/cluster.py?line=769'>770</a>     vector_fields=[self.vector_field],
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/operations/cluster/cluster.py?line=770'>771</a>     alias=alias,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/operations/cluster/cluster.py?line=771'>772</a> )["results"]
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/operations/cluster/cluster.py?line=773'>774</a> relevant_centroids = [
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/operations/cluster/cluster.py?line=774'>775</a>     centroid["centroid_vector"]
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/operations/cluster/cluster.py?line=775'>776</a>     for centroid in centroid_documents
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/operations/cluster/cluster.py?line=776'>777</a>     if any(f"-{cluster}" in centroid["_id"] for cluster in cluster_labels)
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/operations/cluster/cluster.py?line=777'>778</a> ]
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/operations/cluster/cluster.py?line=778'>779</a> new_centroid = np.array(relevant_centroids).mean(0).tolist()


    File c:\users\joseph\documents\jobs\relevance ai\relevanceai\relevanceai\_api\endpoints\services\centroids.py:122, in CentroidsClient.list(self, dataset_id, vector_fields, cluster_ids, alias, page_size, cursor, page, include_vector, similarity_metric)
         <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=94'>95</a> """
         <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=95'>96</a> Retrieve the cluster centroids by IDs
         <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=96'>97</a>
       (...)
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=117'>118</a>     Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=118'>119</a> """
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=119'>120</a> cluster_ids = [] if cluster_ids is None else cluster_ids
    --> <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=121'>122</a> return self.make_http_request(
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=122'>123</a>     f"/datasets/{dataset_id}/cluster/centroids/documents",
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=123'>124</a>     method="POST",
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=124'>125</a>     parameters={
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=125'>126</a>         # "dataset_id": dataset_id,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=126'>127</a>         "cluster_ids": cluster_ids,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=127'>128</a>         "vector_fields": vector_fields,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=128'>129</a>         "alias": alias,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=129'>130</a>         "page_size": page_size,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=130'>131</a>         "cursor": cursor,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=131'>132</a>         "page": page,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=132'>133</a>         "include_vector": include_vector,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=133'>134</a>         "similarity_metric": similarity_metric,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=134'>135</a>         "vector_field": "",
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=135'>136</a>     },
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/_api/endpoints/services/centroids.py?line=136'>137</a> )


    File c:\users\joseph\documents\jobs\relevance ai\relevanceai\relevanceai\utils\transport.py:248, in Transport.make_http_request(self, endpoint, method, parameters, base_url, output_format, raise_error)
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=240'>241</a>     self._log_response_fail(
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=241'>242</a>         base_url,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=242'>243</a>         endpoint,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=243'>244</a>         response.status_code,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=244'>245</a>         response.content.decode(),
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=245'>246</a>     )
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=246'>247</a>     if raise_error:
    --> <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=247'>248</a>         raise APIError(response.content.decode())
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=249'>250</a> # Retry other errors
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=250'>251</a> else:
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=251'>252</a>     self._log_response_fail(
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=252'>253</a>         base_url,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=253'>254</a>         endpoint,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=254'>255</a>         response.status_code,
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=255'>256</a>         response.content.decode(),
        <a href='file:///c%3A/users/joseph/documents/jobs/relevance%20ai/relevanceai/relevanceai/utils/transport.py?line=256'>257</a>     )


    APIError: {"message":"No centroids were found for chosen vector fields and alias."}
