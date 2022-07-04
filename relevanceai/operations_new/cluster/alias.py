import warnings
from typing import Any, Optional
from relevanceai.constants.warning import Warning


class ClusterAlias:
    """Utilities for determining a cluster alias"""

    model: Any
    model_kwargs: dict
    model_name: str
    verbose: bool = False

    def _get_alias(self, alias: Any) -> str:
        # Return the alias
        if alias is not None:
            self.alias = alias
            return alias
        alias = self._get_alias_from_package()

        if alias is not None and isinstance(alias, str):
            return alias
        alias = self._generate_alias()
        return alias.lower()

    def _get_alias_from_package(self):
        self._get_package_from_model(self.model)
        if self.package == "sklearn":
            self.alias = self._get_alias_from_sklearn()
            if self.alias is not None:
                return self.alias

    def _get_alias_from_sklearn(self):
        if hasattr(self.model, "name"):
            if hasattr(self.model, "n_clusters"):
                return f"{self.model.name}-{self.model.n_clusters}"
            elif hasattr(self.model, "k"):
                return f"{self.model.name}-{self.model.k}"
            else:
                return f"{self.model.name}"
        else:
            warnings.warn("No alias has been detected - using model type.")
            return str(type(self.model))

    def _get_n_clusters(self):
        if "n_clusters" in self.model_kwargs:
            return self.model_kwargs["n_clusters"]
        elif hasattr(self.model, "n_clusters"):
            return self.model.n_clusters
        elif hasattr(self.model, "k"):
            return self.model.k
        return None

    def _generate_alias(self) -> str:
        # Issue a warning about auto-generated alias
        # We auto-generate certain aliases if the model
        # is a default model like kmeans or community detection
        n_clusters = self._get_n_clusters()
        if hasattr(self.model, "alias"):
            alias = self.model.alias
            if alias is not None:
                return alias

        if n_clusters is not None:
            alias = f"{self.model_name}-{n_clusters}"
        else:
            alias = f"{self.model_name}"

        if self.verbose:
            print(f"The alias is `{alias.lower()}`.")

        Warning.MISSING_ALIAS.format(alias=alias)
        return alias

    def _get_package_from_model(self, model):
        """
        Determine the package for a model.
        This can be useful for checking dependencies.
        This may be used across modules for
        deeper integrations
        """
        # TODO: add support for huggingface integrations
        # such as transformers and sentencetransformers
        model_name = str(model.__class__).lower()
        if "function" in model_name:
            model_name = str(model.__name__)

        if "sklearn" in model_name:
            self.package = "sklearn"

        elif "faiss" in model_name:
            self.package = "faiss"

        elif "hdbscan" in model_name:
            self.package = "hdbscan"

        elif "communitydetection" in model_name:
            self.package = "sentence-transformers"

        else:
            self.package = "custom"
        return self.package

    def _get_model(self, model: Any, model_kwargs: Optional[dict]) -> Any:
        if model_kwargs is None:
            model_kwargs = {}

        if model is None:
            return model
        if isinstance(model, str):
            model = self._get_model_from_string(model, model_kwargs)

        elif "sklearn" in model.__module__:
            model = self._get_sklearn_model_from_class(model)

        elif "faiss" in model.__module__:
            model = self._get_faiss_model_from_class(model)

        return model

    def _get_sklearn_model_from_class(self, model):
        from relevanceai.operations_new.cluster.models.sklearn.base import (
            SklearnModel,
        )

        model_kwargs = model.__dict__
        model = SklearnModel(model=model, model_kwargs=model_kwargs)
        return model

    def _get_faiss_model_from_class(self, model):
        raise NotImplementedError

    def normalize_model_name(self, model):
        if isinstance(model, str):
            return model.lower().replace("-", "").replace("_", "")

        return model

    def _get_model_from_string(self, model: str, model_kwargs: dict = None):
        if model_kwargs is None:
            model_kwargs = {}

        model = self.normalize_model_name(model)
        model_kwargs = {} if model_kwargs is None else model_kwargs

        from relevanceai.operations_new.cluster.models.sklearn import sklearn_models

        if model in sklearn_models:
            from relevanceai.operations_new.cluster.models.sklearn.base import (
                SklearnModel,
            )

            model = SklearnModel(
                model=model,
                model_kwargs=model_kwargs,
            )
            return model

        elif model == "faiss":
            from relevanceai.operations_new.cluster.models.faiss.base import (
                FaissModel,
            )

            model = FaissModel(
                model=model,
                model_kwargs=model_kwargs,
            )
            return model

        elif model == "communitydetection":
            from relevanceai.operations_new.cluster.models.sentence_transformers.community_detection import (
                CommunityDetectionModel,
            )

            model = CommunityDetectionModel(**model_kwargs)
            return model

        raise ValueError("Model not supported.")
