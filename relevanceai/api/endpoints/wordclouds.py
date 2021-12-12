"""Wordclouds services
"""
from relevanceai.base import _Base


class WordcloudsClient(_Base):
    def __init__(self, project, api_key):
        self.project = project
        self.api_key = api_key
        super().__init__(project, api_key)

    def wordclouds(
        self,
        dataset_id: str,
        fields: list,
        n: int = 2,
        most_common: int = 5,
        page_size: int = 20,
        select_fields: list = [],
        include_vector: bool = False,
        filters: list = [],
        additional_stopwords: list = [],
    ):
        """
        Get frequency n-gram frequency counter from the wordcloud.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        fields: list
            The field on which to build NGrams
        n: int
            The number of words fo combine
        most_common: int
            The most common number of n-gram terms
        page_size: int
            Size of each page of results
        select_fields : list
            Fields to include in the search results, empty array/list means all fields.
        include_vector: bool
            Include vectors in the search results
        filters: list
            Query for filtering the search results
        additional_stopwords: list
            Additional stopwords to add
        """

        return self.make_http_request(
            f"/services/wordclouds/wordclouds",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "fields": fields,
                "n": n,
                "most_common": most_common,
                "page_size": page_size,
                "select_fields": select_fields,
                "include_vector": include_vector,
                "filters": filters,
                "additional_stopwords": additional_stopwords,
            },
        )
