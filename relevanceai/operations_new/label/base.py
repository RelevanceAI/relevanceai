"""
Base operations for labels
"""
from typing import Callable, Dict, List, Optional
from doc_utils import DocUtils

class LabelBase(DocUtils):
    def run(self, 
        vector_field: str,
        documents, label_documents,
        max_number_of_labels: int=1,
        expanded: bool=True,
        similarity_metric: str="cosine",
        similarity_threshold: float=0.1,
        label_field="label", label_vector_field="label_vector_",
        ):
        """
        Labels a given set of documents against
        label documents
        """
        # for each document
        # get vector 
        # match vector against label vectors 
        # store labels
        # return labelled documents
        
        # Get all vectors
        vectors = self.get_field_across_documents(vector_field, documents)
        for v in vectors:
            # search across
            labels = self._get_nearest_labels(v, label_field, label_documents,
                expanded=expanded)
            # TODO: add inplace=True
            self.set_field_across_documents("_label_", labels, documents,)
        return documents
    
    def _get_nearest_labels(self, vector, label_field, label_documents, expanded=True):
        """
        Get the nearest labels
        """
        if expanded:
            return self._get_nearest_labels_expanded(
                vector, label_field, label_documents
            )
        else:
            # anticipate common mistakes
            raise NotImplementedError("please set expanded=True") 

    def _get_nearest_labels_expanded(
        self, vector, label_documents, label_field: str="label"
    ):
        # get the label vectors
        labels = self.get_field(label_field, label_documents)
        return labels
