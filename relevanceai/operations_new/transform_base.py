"""
Base Operations Class
Checks fields, contains key abstract methods, etc
"""
from abc import ABC, abstractmethod

from typing import Any, Dict, List, Union

from relevanceai.utils import DocUtils
from relevanceai.client import Credentials

DO_NOT_STORE = ["_centroids", Credentials.__slots__]


class OperationsCheck(ABC, DocUtils):
    def _check_alias(self):
        if self.alias is None:
            raise ValueError("alias is not set. Please supply alias=")
        if self.alias.lower() != self.alias:
            raise ValueError("Alias cannot be lower case.")

    def _check_vector_fields(self):
        """It checks that the vector fields are in the schema, and that they are of the correct type"""
        self._check_vector_field_type()

        if len(self.vector_fields) == 0:
            raise ValueError("No vector fields set. Please supply with vector_fields=")
        self._check_vector_field_names()
        self._check_vector_field_in_schema()

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

    def _check_vector_field_names(self):
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
                    assert vector_field in self.datasets.schema(self.dataset_id)

    @staticmethod
    def normalize_string(string: str):
        return string.lower().replace("-", "").replace("_", "")

    def _check_fields_in_schema(
        self,
        schema: Dict[str, str],
        fields: List[Union[str, None]],
    ) -> None:
        # Check fields in schema
        if fields is not None:
            for field in fields:
                if field not in schema:
                    raise ValueError(f"{field} not in Dataset schema")


class TransformBase(OperationsCheck):
    """
    To write your own operation, you need to add:
    - name
    - transform
    """

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

        values = {k: v for k, v in self.__dict__.items() if k not in DO_NOT_STORE}

        return dict(
            operation=self.name,
            values=values,
        )

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

    def _generate_output_field(self, field):
        return f"_{self.name}_.{field.lower().replace(' ', '_')}"

    def get_transformers_device(self, device: int = None):
        """
        Automatically returns a GPU device if there is one. Otherwise,
        returns a CPU device for transformers
        """
        if device is not None:
            return device
        import torch

        if torch.cuda.is_available():
            return 0
        return -1
