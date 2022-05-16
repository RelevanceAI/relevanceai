from typing import Optional
from relevanceai.dataset.write import Write


class Operations(Write):
    def label(
        self,
        vector_field: str,
        label_documents,
        expanded=True,
        max_number_of_labels: int = 1,
        similarity_metric: str = "cosine",
        filters: Optional[list] = None,
        chunksize: int = 100,
        similarity_threshold: float = 0,
        label_field: str = "label",
        label_vector_field="label_vector_",
    ):
        """
        This function takes a list of documents and a list of labels, and for each document, it finds the
        most similar label and adds it to the document

        Parameters
        ----------
        vector_field : str
            The name of the field that contains the vector representation of the document.
        label_documents
            A list of documents that contain the labels.
        expanded, optional
            If True, the label_field will be an array of labels. If False, the label_field will be a single
        label.
        max_number_of_labels : int, optional
            The maximum number of labels to assign to each document.
        similarity_metric : str, optional
            The metric used to calculate the similarity between the document and the label.
        filters : Optional[list]
            list of filters to apply to the documents
        chunksize : int, optional
            The number of documents to process at a time.
        similarity_threshold : float, optional
            float=0,
        label_field : str, optional
            The field in the document that will contain the label.
        label_vector_field, optional
            The field in the document that will contain the label vector.

        Returns
        -------
            A list of documents

        """
        from relevanceai.operations_new.label.ops import LabelOps

        ops = LabelOps()
        for documents in self.chunk_dataset(
            select_fields=[vector_field], filters=filters, chunksize=chunksize
        ):
            documents = ops.run(
                vector_field=vector_field,
                documents=documents,
                label_documents=label_documents,
                expanded=expanded,
                max_number_of_labels=max_number_of_labels,
                similarity_metric=similarity_metric,
                similarity_threshold=similarity_threshold,
                label_field=label_field,
                label_vector_field=label_vector_field,
            )
            self.upsert_documents(
                documents,
            )
        return
