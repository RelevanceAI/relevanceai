"""Prediction services
"""
from relevanceai.base import _Base


class PredictionClient(_Base):
    def __init__(self, project, api_key):
        self.project = project
        self.api_key = api_key
        super().__init__(project, api_key)

    def KNN(
        self,
        dataset_id: str,
        vector: list,
        vector_field: str,
        target_field: str,
        k: int = 5,
        weighting: bool or list = True,
        impute_value: int = 0,
        predict_operation: str = "most_frequent",
        include_search_results: bool = True,
    ):

        """
        Predict using KNN regression.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        vector: list
            Vector, a list/array of floats that represents a piece of data.
        vector_field: string
            The vector field to search in. It can either be an array of strings (automatically equally weighted) (e.g. ['check_vector_', 'yellow_vector_']) or it is a dictionary mapping field to float where the weighting is explicitly specified (e.g. {'check_vector_': 0.2, 'yellow_vector_': 0.5})
        target_field: string
            The field to perform regression on.
        k: int
            The number of results for KNN.
        weighting : bool/list
            The weighting for each prediction
        impute_value: int
            The value used to fill in the document when the data is missing.
        predict_operation: string
            How to predict using the vectors. One of most_frequent or `sum_scores
        include_search_results: bool
            Whether to include search results.
        """

        return self.make_http_request(
            f"/services/prediction/regression/knn",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "vector": vector,
                "vector_field": vector_field,
                "target_field": target_field,
                "k": k,
                "weighting": weighting,
                "impute_value": impute_value,
                "predict_operation": predict_operation,
                "include_search_results": include_search_results,
            },
        )

    def KNN_from_results(
        self,
        field: str,
        results: list,
        impute_value: int = 0,
        predict_operation: str = "most_frequent",
    ):

        """
        Predict using KNN regression from search results

        Parameters
        ----------
        field: string
            Field in results to use for the prediction. Can be multiplied with weighting.
        results: dict
            List of results in a dictionary
        weighting : bool/list
            The weighting for each prediction
        impute_value: int
            The value used to fill in the document when the data is missing.
        predict_operation: string
            How to predict using the vectors. One of most_frequent or `sum_scores
        """

        return self.make_http_request(
            f"/services/prediction/regression/knn_from_results",
            method="POST",
            parameters={
                "field": field,
                "results": results,
                "impute_value": impute_value,
                "predict_operation": predict_operation,
            },
        )
