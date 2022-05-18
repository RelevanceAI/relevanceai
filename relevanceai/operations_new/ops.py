from typing import Any, Dict, List, Optional
from relevanceai.dataset.write import Write
from datetime import datetime


class Operations(Write):
    def store_operation_metadata(self, operation: str, values: str):
        """
        Store metadata about operators
        {
            "_operationhistory_": {
                "1-1-1-17-2-3": {
                    "operation": "vector", "model_name": "miniLm"
                },
            }
        }

        """
        print("Storing operation metadata...")
        timestamp = str(datetime.now().timestamp()).replace(".", "-")
        metadata = {
            "_operationhistory_": {
                timestamp: {"operation": operation, "parameters": values}
            }
        }
        # Gets metadata and appends to the operation history
        return self.upsert_metadata(metadata)

    def dim_reduction(
        self,
        fields: List[str],
        models: Optional[List[Any]] = None,
        filters: Optional[List[Dict[str, Any]]] = None,
        chunksize: int = 100,
    ):
        """It takes a list of fields, a list of models, a list of filters, and a chunksize, and then runs
        the DimReductionOps class on the documents in the dataset

        Parameters
        ----------
        fields : List[str]
            List[str]
        models : Optional[List[Any]]
            List[Any] = None,
        filters : Optional[List[Dict[str, Any]]]
            A list of dictionaries, each dictionary containing a filter.
        chunksize : int, optional
            The number of documents to process at a time.

        Returns
        -------
            Nothing is being returned.

        """
        from relevanceai.operations_new.dim_reduction.ops import DimReductionOps

        models = ["pca"] if models is None else models

        ops = DimReductionOps(fields=fields, models=models)
        for documents in self.chunk_dataset(
            select_fields=fields, filters=filters, chunksize=chunksize
        ):
            updated_documents = ops.run(documents)
            self.upsert_documents(
                updated_documents,
            )

        return

    def vectorize_text(
        self,
        fields: List[str],
        models: Optional[List[Any]] = None,
        filters: Optional[List[Dict[str, Any]]] = None,
        chunksize: int = 100,
    ):
        """It takes a list of fields, a list of models, a list of filters, and a chunksize, and then it runs
        the VectorizeOps function on the documents in the database

        Parameters
        ----------
        fields : List[str]
            List[str]
        models : List[Any]
            List[Any]
        filters : List[Dict[str, Any]]
            List[Dict[str, Any]]
        chunksize : int, optional
            int = 100,

        Returns
        -------
            Nothing

        """
        from relevanceai.operations_new.vectorize.text.ops import VectorizeTextOps

        models = ["all-mpnet-base-v2"] if models is None else models

        ops = VectorizeTextOps(fields=fields, models=models)
        for documents in self.chunk_dataset(
            select_fields=fields, filters=filters, chunksize=chunksize
        ):
            updated_documents = ops.run(documents)
            self.upsert_documents(
                updated_documents,
            )
        self.store_operation_metadata(
            operation="vectorize_text",
            values=str({"fields": fields, "models": models, "filters": filters}),
        )
        return

    def vectorize_image(
        self,
        fields: List[str],
        models: List[Any],
        filters: Optional[List[Dict[str, Any]]] = None,
        chunksize: int = 100,
    ):
        """It takes a list of fields, a list of models, a list of filters, and a chunksize, and then it runs
        the VectorizeOps function on the documents in the database

        Parameters
        ----------
        fields : List[str]
            List[str]
        models : List[Any]
            List[Any]
        filters : List[Dict[str, Any]]
            List[Dict[str, Any]]
        chunksize : int, optional
            int = 100,

        Returns
        -------
            Nothing

        """
        from relevanceai.operations_new.vectorize.image.ops import VectorizeImageOps

        ops = VectorizeImageOps(fields=fields, models=models)
        for documents in self.chunk_dataset(
            select_fields=fields, filters=filters, chunksize=chunksize
        ):
            updated_documents = ops.run(documents)
            self.upsert_documents(
                updated_documents,
            )

        self.store_operation_metadata(
            operation="vectorize_image",
            values=str({"fields": fields, "models": models, "filters": filters}),
        )
        return

    def label(
        self,
        vector_fields: List[str],
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
        """This function takes a list of documents, a list of labels, and a list of vector fields, and
        returns a list of documents with the labels added to the documents

        Parameters
        ----------
        vector_fields : List[str]
            List[str]
        label_documents
            A list of documents that contain the labels.
        expanded, optional
            If True, the label_vector_field will be a list of vectors, one for each label. If False, the
        label_vector_field will be a single vector, the sum of all the label vectors.
        max_number_of_labels : int, optional
            int = 1,
        similarity_metric : str, optional
            str = "cosine",
        filters : Optional[list]
            Optional[list] = None,
        chunksize : int, optional
            The number of documents to process at a time.
        similarity_threshold : float, optional
            float = 0,
        label_field : str, optional
            The name of the field that will contain the label.
        label_vector_field, optional
            The field that will contain the label vector.

        Returns
        -------
            The return value is a list of documents.

        """
        from relevanceai.operations_new.label.ops import LabelOps

        ops = LabelOps()
        for documents in self.chunk_dataset(
            select_fields=vector_fields, filters=filters, chunksize=chunksize
        ):
            updated_documents = ops.run(
                vector_field=vector_fields[0],
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
                updated_documents,
            )

        self.store_operation_metadata(
            operation="vectorize_image",
            values=str(
                {
                    "vector_fields": vector_fields,
                    "expanded": expanded,
                    "max_number_of_labels": max_number_of_labels,
                    "similarity_metric": similarity_metric,
                    "filters": filters,
                    "chunksize": chunksize,
                    "similarity_threshold": similarity_threshold,
                    "label_field": label_field,
                    "label_vector_field": label_vector_field,
                    "label_documents": label_documents,
                }
            ),
        )
        return
