"""
Base operations for labels
"""
from typing import Callable, Dict, List, Optional
from doc_utils import DocUtils

class LabelBase(DocUtils):
    def label_documents(
        self, 
        vector_field: str,
        documents, 
        label_documents,
        max_number_of_labels: int=1,
        expanded: bool=True,
        similarity_metric: str="cosine",
        similarity_threshold: float=0.1,
        label_field="label", label_vector_field="label_vector_",
        ):
        '''
        For each document, get the vector, match the vector against label vectors, store labels, return
        labelled documents
        
        Parameters
        ----------
        vector_field : str
            the field in the documents that contains the vector
        documents
            the documents to label
        label_documents
            the documents that contain the labels
        max_number_of_labels : int, optional
            int=1,
        expanded : bool, optional
            if True, then the label vectors are expanded to the same size as the document vectors.
        similarity_metric : str, optional
            str="cosine",
        similarity_threshold : float
            float=0.1,
        label_field, optional
            the field in the label documents that contains the label
        label_vector_field, optional
            the field in the label documents that contains the vector
        
        Returns
        -------
            A list of documents with the field "_label_" set to the list of labels
        
        '''
        # for each document
        # get vector 
        # match vector against label vectors 
        # store labels
        # return labelled documents
        
        # Get all vectors
        vectors = self.get_field_across_documents(vector_field, documents)
        for i, vector in enumerate(vectors):
            # search across
            labels = self._get_nearest_labels(
                vector,
                label_field=label_field,
                label_documents=label_documents,
                expanded=expanded, 
                label_vector_field=label_vector_field
            )
            # TODO: add inplace=True
            self.set_field("_label_", documents[i], labels)
        return documents

    def _get_nearest_labels(
        self,
        vector,
        label_documents,
        label_field: str="label",
        expanded: bool=True,
        label_vector_field: str="label_vector_",
        similarity_metric: str="cosine",
    ):
        # perform cosine similarity
        if similarity_metric == 'cosine':
            labels = self.cosine_similarity(
                query_vector=vector,
                vector_field=label_vector_field,
                documents=label_documents
            )
        else:
            raise ValueError("Only cosine similarity metric is supported at the moment.")

        # for the label vectors
        if expanded:
            return labels
        else:
            # anticipate common mistakes
            return self.get_field_across_documents(label_field, labels)
    
    def cosine_similarity(
        self, 
        query_vector, 
        vector_field, 
        documents,
        reverse=True,
        score_field: str = "_label_score",
    ):
        from scipy.spatial import distance
        sort_key = [
            1 - distance.cosine(i, query_vector)
            for i in self.get_field_across_documents(vector_field, documents)
        ]
        self.set_field_across_documents(score_field, sort_key, documents)
        return sorted(documents, reverse=reverse, key=lambda x: x[score_field])
