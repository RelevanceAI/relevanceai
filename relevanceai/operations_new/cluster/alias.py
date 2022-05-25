import warnings
from typing import Any
from relevanceai.constants.warning import Warning


class ClusterAlias:
    """Utilities for determining a cluster alias"""

    model: Any
    model_kwargs: dict
    model_name: str
    verbose: bool = False

    def _get_alias(self, alias: Any) -> str:
        # Depending a package
        # Get the alias
        if alias is not None:
            self.alias = alias
            return alias
        self._get_package_from_model(self.model)
        if self.package == "sklearn":
            self.alias = self._get_alias_from_sklearn()
            if self.alias is not None:
                return self.alias
        if alias is not None and isinstance(alias, str):
            return alias
        alias = self._generate_alias()
        return alias.lower()

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
            return self.model.alias

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
