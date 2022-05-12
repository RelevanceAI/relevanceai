import warnings
from typing import List, Dict, Optional, Any, Union, Callable
from tqdm.auto import tqdm

from relevanceai.client.helpers import Credentials

from relevanceai.dataset.write import Write
from relevanceai.utils.decorators.analytics import track
from relevanceai.operations.vector.vectorizer import Vectorizer
from relevanceai.utils.logger import FileLogger

from relevanceai.dataset.io import IO


class Operations(Write, IO):
    def __init__(self, credentials: Credentials, dataset_id: str, **kwargs):
        self.credentials = credentials
        self.dataset_id = dataset_id
        super().__init__(credentials=self.credentials, dataset_id=dataset_id, **kwargs)

    @track
    def cluster(
        self,
        model: Any = None,
        vector_fields: Optional[List[str]] = None,
        alias: Optional[str] = None,
        cluster_config: Optional[dict] = None,
        include_cluster_report: bool = True,
        **kwargs,
    ):
        """
        Run clustering on your dataset.

        Example
        ----------

        .. code-block::

            from sklearn.cluster import KMeans
            model = KMeans()

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("sample")
            cluster_ops = ds.cluster(
                model=model, vector_fields=["sample_vector_"],
                alias="kmeans-8"
            )

        Parameters
        ------------

        model: Union[str, Any]
            Any K-Means model
        vector_fields: List[str]
            A list of possible vector fields
        alias: str
            The alias to be used to store your model
        cluster_config: dict
            The cluster config to use
            You can change the number of clusters for kmeans using:
            `cluster_config={"n_clusters": 10}`. For a full list of
            possible parameters for different models, simply check how
            the cluster models are instantiated.

        """
        from relevanceai.operations.cluster import ClusterOps

        ops = ClusterOps(
            credentials=self.credentials,
            model=model,
            alias=alias,
            vector_fields=vector_fields,
            cluster_config=cluster_config,
            **kwargs,
        )
        ops(
            dataset_id=self.dataset_id,
            vector_fields=vector_fields,
            include_cluster_report=include_cluster_report,
        )
        if alias is None:
            alias = ops.alias
        print(
            f"You can now utilise the ClusterOps object using `cluster_ops = client.ClusterOps(alias='{alias}', vector_fields={vector_fields}, dataset_id='{self.dataset_id}')`"
        )
        return ops

    @track
    def reduce_dims(
        self,
        alias: str,
        vector_fields: List[str],
        model: Any = "umap",
        n_components: int = 3,
        **kwargs,
    ):
        """
        Reduce dimensions

        Parameters
        --------------

        model: Callable
            model to reduce dimensions
        n_components: int
            The number of components
        alias: str
            The alias of the model
        vector_fields: List[str]
            The list of vector fields to support

        .. code-block::

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("sample")
            ds.reduce_dims(
                alias="sample",
                vector_fields=["sample_1_vector_"],
                model="umap"
            )

        """
        from relevanceai.operations.dr import ReduceDimensionsOps

        ops = ReduceDimensionsOps(
            credentials=self.credentials,
            model=model,
            n_components=n_components,
            **kwargs,
        )
        return ops.run(
            dataset_id=self.dataset_id,
            vector_fields=vector_fields,
            alias=alias,
        )

    dimensionality_reduction = reduce_dims

    @track
    def vectorize(
        self,
        fields: List[str] = None,
        filters: Optional[List] = None,
        **kwargs,
    ):
        """
        Vectorize the model

        Parameters
        ----------
        fields: List[str]
            A list of fields to vectorize

        encoders : Dict[str, List[Any]]
            A dictionary that creates a mapping between your unstructured fields
            and a list of encoders to run over those unstructured fields

        Returns
        -------
        dict
            If the vectorization process is successful, this dict contains
            the added vector names. Else, the dict is the request result
            containing error information.

        Example
        -------
        .. code-block::

            from relevanceai import Client

            client = Client()

            dataset_id = "sample_dataset_id"
            ds = client.Dataset(dataset_id)

            ds.vectorize(
                fields=["text_field_1", "text_field_2"],
                encoders={
                    "text": ["mpnet", "use"]
                }
            )

            # This operation with create 4 new vector fields
            #
            # text_field_1_mpnet_vector_, text_field_1_mpnet_vector_
            # text_field_1_use_vector_, text_field_1_use_vector_

        """
        if filters is None:
            filters = []
        from relevanceai.operations.vector import VectorizeOps

        ops = VectorizeOps(
            credentials=self.credentials,
            **kwargs,
        )

        return ops(
            dataset_id=self.dataset_id,
            fields=[] if fields is None else fields,
            filter=filters,
        )

    def advanced_vectorize(self, vectorizers: List[Vectorizer]):
        """
        Advanced vectorization.
        By setting an

        Example
        ----------

        .. code-block::

            # When first vectorizing
            from relevanceai.operations import Vectorizer
            vectorizer = Vectorizer(field="field_1", model=model, alias="value")
            ds.advanced_vectorize(
                [vectorizer],
            )

        Parameters
        -------------

        vectorize_mapping: dict
            Vectorize mapping

        """
        # TODO: Write test for advanced vectorize
        all_fields = [v.field for v in vectorizers]
        for vectorizer in tqdm(vectorizers):

            def encode(docs):
                docs = vectorizer.encode_documents(
                    fields=[vectorizer.field], documents=docs
                )
                return docs

            self.pull_update_push_async(
                dataset_id=self.dataset_id,
                update_function=encode,
                updating_args=None,
                select_fields=all_fields,
            )

    @track
    def launch_cluster_app(self, configuration: dict = None):
        """
        Launch an app with a given configuration


        Example
        --------

        .. code-block::

            ds.launch_cluster_app()

        Parameters
        -----------

        configuration: dict
            The configuration can be found in the deployable once created.

        """
        if configuration is None:
            url = f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
            print(
                "Build your clustering app here: "
                f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
            )
            return
        if "configuration" in configuration:
            configuration = configuration["configuration"]
        results = self.deployables.create(
            dataset_id=self.dataset_id, configuration=configuration
        )

        # After you have created an app
        url = f"https://cloud.relevance.ai/dataset/{results['dataset_id']}/deploy/cluster/{self.project}/{self.api_key}/{results['deployable_id']}/{self.region}"
        print(f"You can now access your deployable at {url}")
        return url

    @track
    def subcluster(
        self,
        model,
        alias: str,
        vector_fields,
        parent_field,
        filters: Optional[list] = None,
        min_parent_cluster_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Subcluster

        Parameters
        -------------

        min_parent_cluster_size: Optional[int]
            The minium number of cluster data points for it to cluster on.
            If Less than, then it doesn't work.

        """
        from relevanceai.operations.cluster import SubClusterOps

        ops = SubClusterOps(
            model=model,
            credentials=self.credentials,
            alias=alias,
            vector_fields=vector_fields,
            dataset_id=self.dataset_id,
            parent_field=parent_field,
            dataset=self.dataset_id,
            **kwargs,
        )
        return ops.fit_predict(
            dataset=self.dataset_id,
            vector_fields=vector_fields,
            filters=filters,
            min_parent_cluster_size=min_parent_cluster_size,
        )

    @track
    def add_sentiment(
        self,
        field: str,
        output_field: str = None,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
        highlight: bool = False,
        positive_sentiment_name: str = "positive",
        max_number_of_shap_documents: Optional[int] = None,
        min_abs_score: float = 0.1,
        **apply_args,
    ):
        """
        Easily add sentiment to your dataset

        Example
        ----------

        .. code-block::

            ds.add_sentiment(field="sample_1_label")

        Parameters
        --------------

        field: str
            The field to add sentiment to
        output_field: str
            Where to store the sentiment values
        model_name: str
            The HuggingFace Model name.
        log_to_file: bool
            If True, puts the logs in a file. Otherwise, it will
        highlight: bool
            If True, this will include a SHAP explainer of what is causing positive
            and negative sentiment
        max_number_of_shap_documents: int
            The maximum number of shap documents
        min_abs_score: float
            The minimum absolute score for it to be considered important based on SHAP algorithm.

        """
        from relevanceai.operations.text.sentiment import SentimentOps

        if output_field is None:
            output_field = "_sentiment_." + field

        ops = SentimentOps(model_name=model_name)

        def analyze_sentiment(text):
            return ops.analyze_sentiment(
                text=text,
                highlight=highlight,
                positive_sentiment_name=positive_sentiment_name,
                max_number_of_shap_documents=max_number_of_shap_documents,
                min_abs_score=min_abs_score,
            )

        def analyze_sentiment_document(doc):
            self.set_field(output_field, doc, ops.analyze_sentiment(doc.get(field)))
            return doc

        return self.bulk_apply(
            analyze_sentiment_document,
            select_fields=[field],
            **apply_args,
        )

        # return .fit_dataset(
        #     dataset=self,
        #     input_field=field,
        #     output_field=output_field,
        #     log_to_file=log_to_file,
        #     chunksize=chunksize,
        #     workflow_alias=workflow_alias,
        #     notes=notes,
        #     refresh=refresh,
        #     highlight=highlight,
        #     positive_sentiment_name=positive_sentiment_name,
        #     max_number_of_shap_documents=max_number_of_shap_documents,
        #     min_abs_score=min_abs_score,
        # )

    @track
    def question_answer(
        self,
        input_field: str,
        questions: Union[List[str], str],
        output_field: Optional[str] = None,
        model_name: str = "mrm8488/deberta-v3-base-finetuned-squadv2",
        verbose: bool = True,
        log_to_file: bool = True,
        filters: Optional[list] = None,
    ):
        """
        Question your dataset and retrieve answers from it.

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("ecommerce")
            ds.question_answer(
                input_field="product_title",
                question="What brand shoes",
                output_field="_question_test"
            )

        Parameters
        --------------

        field: str
            The field to add sentiment to
        output_field: str
            Where to store the sentiment values
        model_name: str
            The HuggingFace Model name.
        verbose: bool
            If True, prints progress bar workflow
        log_to_file: bool
            If True, puts the logs in a file.

        """
        from relevanceai.workflow.sequential import SequentialWorkflow, Input, Output
        from relevanceai.operations.text.qa.qa import QAOps

        if isinstance(questions, str):
            # Force listing so it loops through multiple question
            questions = [questions]

        model = QAOps(model_name=model_name)

        for question in tqdm(questions):
            print(f"Processing `{question}`...")

            def bulk_question_answer(contexts: list):
                return model.bulk_question_answer(question=question, contexts=contexts)

            if output_field is None:
                output_field = "_question_." + "-".join(
                    question.lower().strip().split()
                )
                print(f"No output field is detected. Setting to {output_field}")

            workflow = SequentialWorkflow(
                list_of_operations=[
                    Input([input_field], filters=filters),
                    bulk_question_answer,
                    Output(output_field),
                ]
            )
            workflow.run(self, verbose=verbose, log_to_file=log_to_file)

    def translate(self, translation_model_name: str):
        raise NotImplementedError

    # def summarize(
    #     self,
    #     summarize_fields: List[str],
    #     model_name: str = "sshleifer/distilbart-cnn-6-6",
    #     verbose: bool = True,
    #     log_to_file: bool = True,
    # ):
    #     """
    #     Question your dataset and retrieve answers from it.

    #     Example
    #     ----------

    #     .. code-block::

    #         from relevanceai import Client
    #         client = Client()
    #         ds = client.Dataset("ecommerce")
    #         ds.question_answer(
    #             input_field="product_title",
    #             question="What brand shoes",
    #             output_field="_question_test"
    #         )

    #     Parameters
    #     --------------

    #     field: str
    #         The field to add sentiment to
    #     output_field: str
    #         Where to store the sentiment values
    #     model_name: str
    #         The HuggingFace Model name.
    #     verbose: bool
    #         If True, prints progress bar workflow
    #     log_to_file: bool
    #         If True, puts the logs in a file.

    #     """
    #     from relevanceai.workflow.sequential import SequentialWorkflow, Input, Output
    #     from relevanceai.operations.text.qa.qa import QAOps

    #     model = QAOps(model_name=model_name)

    #     def bulk_question_answer(contexts: list):
    #         return model.bulk_question_answer(question=question, contexts=contexts)

    #     if output_field is None:
    #         output_field = "_question_." + "-".join(question.lower().strip().split())
    #         print(f"No output field is detected. Setting to {output_field}")

    #     workflow = SequentialWorkflow(
    #         list_of_operations=[
    #             Input([input_field]),
    #             bulk_question_answer,
    #             Output(output_field),
    #         ]
    #     )
    #     return workflow.run(self, verbose=verbose, log_to_file=log_to_file)
    def search(
        self,
        query: str = None,
        vector_search_query: Optional[dict] = None,
        fields_to_search: Optional[List] = None,
        select_fields: Optional[List] = None,
        **kwargs,
    ):
        """
        Advanced Search

        Parameters
        -----------
        query: str
            The query to use
        vector_search_query: dict
            The vector search query
        fields_to_search: list
            The list of fields to search
        select_fields: list
            The fields to select

        """
        return self.datasets.fast_search(
            dataset_id=self.dataset_id,
            query=query,
            vectorSearchQuery=vector_search_query,
            fieldsToSearch=fields_to_search,
            includeFields=select_fields,
            **kwargs,
        )

    @track
    def list_deployables(self):
        """
        Use this function to list available deployables
        """
        return self.deployables.list()

    @track
    def train_text_model_with_gpl(
        self, text_field: str, title_field: Optional[str] = None
    ):
        """
        Train a text model using GPL (Generative Pseudo-Labelling)
        This can be helpful for `domain adaptation`.

        Example
        ---------

        .. code-block::

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("sample")
            ds.train_text_model(method="gpl")

        Parameters
        ------------

        text_field: str
            Text field

        """
        # The model can also be trained using this method
        from relevanceai.operations.text_finetuning import GPLOps

        ops = GPLOps.from_dataset(dataset=self)
        return ops.run(dataset=self, text_field=text_field, title_field=title_field)

    @track
    def train_text_model_with_tripleloss(
        self,
        text_field: str,
        label_field: str,
        output_dir: str = "trained_model",
        percentage_for_dev=None,
    ):
        """
        Supervised training a text model using tripleloss

        Example
        ---------

        .. code-block::
            from relevanceai import Client
            client = Client()
            ds = client.Dataset("ecommerce")
            ops = SupervisedTripleLossFinetuneOps.from_dataset(
                dataset=ds,
                base_model="distilbert-base-uncased",
                batch_size=16,
                triple_loss_type:str='BatchHardSoftMarginTripletLoss'
            )
            ops.operate(text_field="detail_desc", label_field="_cluster_.desc_use_vector_.kmeans-10", output_dir)

        Parameters
        ------------

        text_field: str
            The field you want to use as input text for fine-tuning
        label_field: str
            The field indicating the classes of the input
        output_dir: str
            The path of the output directory
        percentage_for_dev: float
            a number between 0 and 1 showing how much of the data should be used for evaluation. No evaluation if None

        """
        # The model can also be trained using this method
        from relevanceai.operations.text_finetuning import (
            SupervisedTripleLossFinetuneOps,
        )

        ops = SupervisedTripleLossFinetuneOps.from_dataset(dataset=self)
        return ops.run(
            dataset=self,
            text_field=text_field,
            label_field=label_field,
            output_dir=output_dir,
            percentage_for_dev=percentage_for_dev,
        )

    def ClusterOps(self, alias, vector_fields: List, verbose: bool = False, **kwargs):
        """
        ClusterOps object
        """
        from relevanceai import ClusterOps

        return ClusterOps(
            credentials=self.credentials,
            alias=alias,
            vector_fields=vector_fields,
            dataset_id=self.dataset_id,
            verbose=verbose,
            **kwargs,
        )

    @track
    def label_from_list(
        self,
        vector_field: str,
        model: Callable,
        label_list: list,
        similarity_metric="cosine",
        number_of_labels: int = 1,
        score_field: str = "_search_score",
        alias: Optional[str] = None,
    ):
        """Label from a given list.

        Parameters
        ------------

        vector_field: str
            The vector field to label in the original dataset
        model: Callable
            This will take a list of strings and then encode them
        label_list: List
            A list of labels to accept
        similarity_metric: str
            The similarity metric to accept
        number_of_labels: int
            The number of labels to accept
        score_field: str
            What to call the scoring of the labels
        alias: str
            The alias of the labels

        Example
        --------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")

            # Get a model to help us encode
            from vectorhub.encoders.text.tfhub import USE2Vec
            enc = USE2Vec()

            # Use that model to help with encoding
            label_list = ["dog", "cat"]

            df = client.Dataset("_github_repo_vectorai")

            df.label_from_list("documentation_vector_", enc.bulk_encode, label_list, alias="pets")

        """
        if alias is None:
            warnings.warn(
                "No alias is detected for labelling. Default to 'default' as the alias."
            )
            alias = "default"
        print("Encoding labels...")
        label_vectors = []
        for c in self.chunk(label_list, chunksize=20):
            with FileLogger(verbose=True):
                label_vectors.extend(model(c))

        if len(label_vectors) == 0:
            raise ValueError("Failed to encode.")

        # we need this to mock label documents - these values are not important
        # and can be changed :)
        LABEL_VECTOR_FIELD = "label_vector_"
        LABEL_FIELD = "label"

        label_documents = [
            {LABEL_VECTOR_FIELD: label_vectors[i], LABEL_FIELD: label}
            for i, label in enumerate(label_list)
        ]

        return self._bulk_label_dataset(
            label_documents=label_documents,
            vector_field=vector_field,
            label_vector_field=LABEL_VECTOR_FIELD,
            similarity_metric=similarity_metric,
            number_of_labels=number_of_labels,
            score_field=score_field,
            label_fields=[LABEL_FIELD],
            alias=alias,
        )

    def _bulk_label_dataset(
        self,
        label_documents,
        vector_field,
        label_vector_field,
        similarity_metric,
        number_of_labels,
        score_field,
        label_fields,
        alias,
    ):
        def label_and_store(d: dict):
            labels = self._get_nearest_labels(
                label_documents=label_documents,
                vector=self.get_field(vector_field, d),
                label_vector_field=label_vector_field,
                similarity_metric=similarity_metric,
                number_of_labels=number_of_labels,
                score_field=score_field,
                label_fields=label_fields,
            )
            d.update(self._store_labels_in_document(labels, alias))
            return d

        def bulk_label_documents(documents):
            [label_and_store(d) for d in documents]
            return documents

        print("Labelling dataset...")
        return self.bulk_apply(
            bulk_label_documents,
            filters=[
                {
                    "field": vector_field,
                    "filter_type": "exists",
                    "condition": ">=",
                    "condition_value": " ",
                },
            ],
            select_fields=[vector_field],
        )

    @track
    def _store_labels_in_document(self, labels: list, alias: str):
        if isinstance(labels, dict) and "label" in labels:
            return {"_label_": {alias: labels["label"]}}
        return {"_label_": {alias: labels}}

    def _get_nearest_labels(
        self,
        label_documents: List[Dict],
        vector: List[float],
        label_vector_field: str,
        similarity_metric,
        number_of_labels: int,
        label_fields: List[str],
        score_field="_label_score",
    ):

        from relevanceai.operations.vector.local_nearest_neighbours import (
            NearestNeighbours,
        )

        nearest_neighbors: List[Dict] = NearestNeighbours.get_nearest_neighbours(
            label_documents,
            vector,
            label_vector_field,
            similarity_metric,
            score_field=score_field,
        )[:number_of_labels]
        labels: List[Dict] = self.subset_documents(
            [score_field] + label_fields, nearest_neighbors
        )
        # Preprocess labels for easier frontend access
        new_labels = {}
        for lf in label_fields:
            new_labels[lf] = [
                {"label": l.get(lf), score_field: l.get(score_field)} for l in labels
            ]
        return new_labels
