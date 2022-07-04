"""

"""
from tqdm.auto import tqdm
from typing import Optional
from relevanceai.operations_new.cluster.text.explainer.base import BaseExplainer
from relevanceai.operations_new.ops_base import OperationAPIBase


class TextClusterExplainerOps(BaseExplainer, OperationAPIBase):  # type: ignore
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, *args, **kwargs):
        # We transform isn't very useful here - it's mainly used for explain_clusters
        return self.explain_clusters(*args, **kwargs)

    def explain_clusters(
        self,
        dataset_id: str,
        vector_fields: list,
        alias: str,
        text_field,
        encode_fn,
        n_closest: int = 5,
        highlight_output_field="_explain_",
    ):

        """
        For each cluster, we explain the closest documents to the centroid, and then highlight the most
        important words based on what's most similar to the cluster center

        Parameters
        ----------
        text_field
            The field that contains the text to explain
        encode_fn
            The function that encodes the text. This is the same function that you used to encode the text in
        the dataset.
        n_closest : int, optional
            The number of closest documents to the centroid to explain
        highlight_output_field, optional
            The field that will be added to the document with the highlighted text.

        .. code-block::

            from relevanceai import Client
            client = Client()


        """
        # TODO:
        # Make this supported with cluster_ops.explain_text_clusters()

        # For each centroid, explains the closest n_closest to the centroids
        # Then highlights why the centroid is similar to the second most similar one
        # List the closest
        if len(vector_fields) > 1:
            raise NotImplementedError(
                "Currently do not support more than 1 vector fields."
            )

        from relevanceai.operations_new.cluster.ops import ClusterOps

        cluster_ops = ClusterOps(
            credentials=self.credentials,
            vector_fields=vector_fields,
            alias=alias,
            dataset_id=dataset_id,
            model=None,
        )
        # get get the centroid from the centroid
        closest = cluster_ops.list_closest(
            page_size=n_closest, select_fields=[text_field], verbose=False
        )
        # For the results in closest, we explain a text field
        # Flatten the closest
        for cluster, results in tqdm(closest["results"].items()):
            centroid_document = cluster_ops.get_centroid_from_id(cluster)
            query_vector = centroid_document[vector_fields[0]]

            if len(results["results"]) < 1:
                continue

            result_texts = self.get_field_across_documents(
                text_field, results["results"]
            )
            # Explain the first one
            for i, r in enumerate(result_texts):
                explained_answer = self.explain_from_vector(
                    encode_fn, query_vector=query_vector, answer_text=r
                )
                results["results"][i][highlight_output_field] = {
                    text_field: explained_answer
                }

            # Update explained results
            updated_results = self._update_documents(
                dataset_id=cluster_ops.dataset_id, documents=results["results"]
            )

        print(
            f"Make sure to set the highlight field `{text_field}` with substring `{highlight_output_field + '.' + text_field}` at: "
        )
        print(
            f"https://cloud.relevance.ai/dataset/{cluster_ops.dataset_id}/dashboard/settings"
        )
        return closest

    @property
    def name(self):
        return "text-explainer"

    def run(self):
        pass

    def explain_clusters_relational(
        self,
        dataset_id: str,
        vector_fields: list,
        alias: str,
        text_field,
        encode_fn,
        n_closest: int = 5,
        highlight_output_field="_explain_",
    ):

        """
        For each cluster, we explain the closest documents to the centroid, and then highlight the most
        important words in the centroid and then for the centroid, we analyse what makes it similar to the second most similar document

        Parameters
        ----------
        cluster_ops : ClusterOps
            ClusterOps,
        text_field
            The field that contains the text to explain
        encode_fn
            The function that encodes the text. This is the same function that you used to encode the text in
        the dataset.
        n_closest : int, optional
            The number of closest documents to the centroid to explain
        highlight_output_field, optional
            The field that will be added to the document with the highlighted text.
        .. code-block::
            from relevanceai import Client
            client = Client()
        """
        # TODO:
        # Make this supported with cluster_ops.explain_text_clusters()

        # For each centroid, explains the closest n_closest to the centroids
        # Then highlights why the centroid is similar to the second most similar one
        # List the closest

        if len(vector_fields) > 1:
            raise NotImplementedError(
                "Currently do not support more than 1 vector fields."
            )

        from relevanceai.operations_new.cluster.ops import ClusterOps

        cluster_ops = ClusterOps(
            credentials=self.credentials,
            vector_fields=vector_fields,
            alias=alias,
            dataset_id=dataset_id,
            model=None,
        )

        closest = cluster_ops.list_closest(
            page_size=n_closest, select_fields=[text_field], verbose=False
        )
        # For the results in closest, we explain a text field
        # Flatten the closest
        for cluster, results in tqdm(closest["results"].items()):
            if len(results["results"]) < 2:
                continue
            query_text = self.get_field(text_field, results["results"][0])
            result_texts = self.get_field_across_documents(
                text_field, results["results"][1:]
            )
            # Explain the first one
            explained_answer = self.explain(
                encode_fn, query_text=result_texts[0], answer_text=query_text
            )

            # results["results"][0][highlight_output_field] = {
            #     text_field: explained_answer
            # }
            self.set_field(
                highlight_output_field + "." + text_field,
                results["results"][0],
                explained_answer,
            )
            for i, r in enumerate(result_texts):
                explained_answer = self.explain(
                    encode_fn, query_text=query_text, answer_text=r
                )
                split_text_fields = text_field.split(".")
                # Here we want to store nested fields
                # For example "value.check": "this is answer"
                # should be stored like this:
                # {"value": {"check": "this is answer"}}
                self.set_field(
                    f"{highlight_output_field}.{text_field}",
                    results["results"][i + 1],
                    explained_answer,
                )

            # Update explained results
            updated = self._update_documents(
                dataset_id=cluster_ops.dataset_id, documents=results["results"]
            )

        print(
            f"Make sure to set the highlight field `{text_field}` with substring `{highlight_output_field + '.' + text_field}` at: "
        )
        print(
            f"https://cloud.relevance.ai/dataset/{cluster_ops.dataset_id}/dashboard/settings"
        )
        self.datasets.post_settings(
            dataset_id=cluster_ops.dataset_id,
            settings={
                "settings": {
                    "highlightingRules": [
                        {
                            "substringField": highlight_output_field,
                            "weightField": "",
                            "fullField": text_field,
                        }
                    ]
                }
            },
        )
        return closest
