from typing import List, Union

import numpy as np

from relevanceai.operations_new.dr.models.base import DimReductionModelBase

try:
    from ivis import Ivis
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{e}\nInstall ivis\n \
        CPU: pip install -U ivis[cpu]\n \
        GPU: pip install -U ivis[gpu]"
    )


class IvisModel(DimReductionModelBase):
    def __init__(
        self,
        n_components: int,
        alias: Union[str, None],
        **kwargs,
    ):
        self.model = Ivis(embedding_dims=n_components, **kwargs)
        self.model_name = "ivis"
        self.alias = alias

    def fit(
        self,
        vectors: Union[List[List[float]], np.ndarray],
    ) -> None:
        """The function takes a list of lists of floats and fits the model to the data

        Parameters
        ----------
        vectors : Union[List[List[float]], np.ndarray]
            Union[List[List[float]], np.ndarray]

        """

        if isinstance(vectors, list):
            vectors = np.array(vectors)

        if self.model.batch_size > vectors.shape[0]:
            self.model.batch_size = vectors.shape[0]

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
        self.model.fit(vectors)
        reduce_vectors = self.model.transform(vectors)
        return reduce_vectors.tolist()
