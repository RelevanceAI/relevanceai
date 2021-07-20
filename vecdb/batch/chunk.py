"""Chunk Helper functions
"""
import pandas as pd
from typing import Union, List
from ..progress_bar import progress_bar

class Chunker:
    """Update the chunk mixins
    """
    def chunk(self, documents: Union[pd.DataFrame, List], chunksize: int=20):
        """
        Chunk an iterable object in Python.
        Args:
            documents:
                List of dictionaries/Pandas dataframe
            chunksize:
                The chunk size of an object.
        Example:
            >>> documents = [{...}]
            >>> ViClient.chunk(documents)
        """
        if isinstance(documents, pd.DataFrame):
            for i in progress_bar(range(0, len(documents) / chunksize)):
                yield documents.iloc[i * chunksize : (i + 1) * chunksize]
        else:
            for i in progress_bar(range(0, int(len(documents) / chunksize))):
                yield documents[i * chunksize : ((i + 1)*chunksize)]
