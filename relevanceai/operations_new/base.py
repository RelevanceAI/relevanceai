"""
Base Operations Class
"""
from abc import ABC, abstractmethod

from typing import Any, Dict, List

from relevanceai.utils import DocUtils


class OperationBase(ABC, DocUtils):

    # Typehints to help with development
    vector_fields: List[str]
    alias: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.transform(*args, **kwargs)

    @property
    @abstractmethod
    def name(self) -> str:
        """abstractproperty for name"""
        raise NotImplementedError

    @abstractmethod
    def transform(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """abstractmethod for transform"""
        raise NotImplementedError

    def get_operation_metadata(self, *args, **kwargs) -> Dict[str, Any]:
        """abstractmethod for return metadata for upsertion"""
        return dict(
            operation=self.name,
            values=self.__dict__,
        )

    def _check_vector_field_type(self):
        """If the vector_fields is None, raise an error. If it's a string, force it to be a list"""
        # check the vector fields
        if self.vector_fields is None:
            raise ValueError(
                "No vector_fields has been set. Please supply with vector_fields="
            )
        elif isinstance(self.vector_fields, str):
            # Force it to be a list instead
            self.vector_fields = [self.vector_fields]

    def _check_vector_names(self):
        """If the class has a vector_fields attribute, then for each vector field in the vector_fields
        attribute, check that the vector field ends with _vector_. If it doesn't, raise a ValueError

        """
        if hasattr(self, "vector_fields"):
            for vector_field in self.vector_fields:
                if not vector_field.endswith("_vector_"):
                    raise ValueError(
                        "Invalid vector field. Ensure they end in `_vector_`."
                    )

    def _check_vector_field_in_schema(self):
        """> If the dataset has a schema, then the vector field must be in the schema"""
        # TODO: check the schema
        if hasattr(self, "dataset"):
            for vector_field in self.vector_fields:
                if hasattr(self.dataset, "schema"):
                    assert vector_field in self.dataset.schema

    def _check_vector_fields(self):
        """It checks that the vector fields are in the schema, and that they are of the correct type"""
        self._check_vector_field_type()

        if len(self.vector_fields) == 0:
            raise ValueError("No vector fields set. Please supply with vector_fields=")
        self._check_vector_names()
        self._check_vector_field_in_schema()

    def _check_alias(self):
        if self.alias is None:
            raise ValueError("alias is not set. Please supply alias=")
        if self.alias.lower() != self.alias:
            raise ValueError("Alias cannot be lower case.")

    def _get_package_from_model(self, model):
        """Determine the package for a model.
        This can be useful for checking dependencies.
        This may be used across modules for
        deeper integrations

        Parameters
        ----------
        model
            The model to be used for clustering.

        Returns
        -------
            The package name

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

    @staticmethod
    def normalize_string(string: str):
        return string.lower().replace("-", "").replace("_", "")
