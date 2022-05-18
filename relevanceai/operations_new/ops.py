from typing import Any, Dict, List, Optional
from relevanceai.dataset.write import Write


class Operations(Write):
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
        return
