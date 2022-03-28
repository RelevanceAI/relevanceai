"""Chunk Helper functions"""
from typing import List, Union

import pandas as pd
import math

from relevanceai.utils.progress_bar import progress_bar


class Chunker:
    """Update the chunk Mixins"""

    def chunk(self, documents: Union[pd.DataFrame, List], chunksize: int = 20):
        """
        Chunk an iterable object in Python. \n
        Example:

        >>> documents = [{...}]
        >>> ViClient.chunk(documents)

        Parameters
        ----------
        documents:
            List of dictionaries/Pandas dataframe
        chunksize:
            The chunk size of an object.
        """
        if isinstance(documents, pd.DataFrame):
            for i in progress_bar(range(0, math.ceil(len(documents) / chunksize))):
                yield documents.iloc[i * chunksize : (i + 1) * chunksize]
        else:
            for i in progress_bar(range(0, math.ceil(len(documents) / chunksize))):
                yield documents[i * chunksize : ((i + 1) * chunksize)]
