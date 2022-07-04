from typing import List, Union, Optional

import numpy as np

from relevanceai.operations_new.scaling.models.base import ScalerModelBase

from sklearn.base import TransformerMixin

from relevanceai.operations_new.scaling.models import sklearn_models


class SKLearnScaler(ScalerModelBase):
    def __init__(
        self,
        model: Union[str, TransformerMixin],
        alias: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        self.alias = alias
        self.model_kwargs = model_kwargs
        self.model: TransformerMixin = self._get_model(model)
        self.model_name = type(self.model).__name__.lower()

    def _get_model(self, model: Union[str, TransformerMixin]):

        if isinstance(model, str):
            model_string = sklearn_models[model]
            model = ScalerModelBase.import_from_string(
                f"sklearn.preprocessing.{model_string}"
            )

            model = model(**self.model_kwargs)

        return model

    def fit(
        self,
        vectors: Union[List[List[float]], np.ndarray],
    ) -> None:
        """It fits the model to the vectors.

        Parameters
        ----------
        vectors : Union[List[List[float]], np.ndarray]
            Union[List[List[float]], np.ndarray]

        """

        if isinstance(vectors, list):
            vectors = np.array(vectors)

        self.model.fit(vectors)

    def fit_transform(
        self,
        vectors: List[List[float]],
    ) -> List[List[float]]:
        """It takes a list of vectors, fits the model to the vectors, and then transforms the vectors

        Parameters
        ----------
        vectors : List[List[float]]
            List[List[float]]

        Returns
        -------
            A list of lists of floats.

        """

        vectors = np.array(vectors)
        reduced_vectors = self.model.fit_transform(vectors)
        return reduced_vectors.tolist()
