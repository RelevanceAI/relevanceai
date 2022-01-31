from relevanceai.dataset_api.dataset_write import Write


class DR(Write):
    """Relevance AI offers quick and easy dimensionality reduction"""

    def reduce_dimensions(
        self,
        vector_fields: list,
        alias: str,
        number_of_documents: int = 1000,
        algorithm: str = "pca",
        n_components: int = 3,
        filters: list = [],
    ):
        """
        Run dimensionality reduction quickly on a dataset on a small number of documents.
        This is useful if you want to quickly see a projection of your dataset.
        Currently, the only supported algorithm is `PCA`.

        .. warning::
            This function is currently in beta and is likely to change in the future.
            We recommend not using this in any production systems.

        Parameters
        ----------
        vector_fields: list
            The vector fields to run dimensionality reduction on
        number_of_documents: int
            The number of documents to get
        algorithm: str
            The algorithm to run. The only supported algorithm is `pca` at this
            current point in time.
        n_components: int
            The number of components

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")
            df.reduce_dimensions(
                ["sample_vector_"],
                alias="pca",
                number_of_documents=1000
            )

        """
        if len(vector_fields) > 1:
            raise ValueError("We only support 1 vector field at the moment.")

        print("Getting documents...")
        documents = self.get_documents(
            dataset_id=self.dataset_id,
            select_fields=vector_fields,
            filters=filters,
            number_of_documents=number_of_documents,
        )

        print("Run PCA...")
        if algorithm == "pca":
            dr_docs = self._run_pca(
                vector_fields=vector_fields,
                documents=documents,
                alias=alias,
                n_components=n_components,
            )
        else:
            raise ValueError("DR algorithm not supported.")

        return self.update_documents(self.dataset_id, dr_docs)

    def _run_pca(
        self, vector_fields: list, documents: list, alias: str, n_components: int = 3
    ):
        from relevanceai.vector_tools.dim_reduction import PCA

        model = PCA()
        # Returns a list of documents with the dr vector
        return model.fit_transform_documents(
            vector_field=vector_fields[0], documents=documents, alias=alias
        )
