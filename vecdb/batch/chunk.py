"""Chunk Helper functions
"""
import pandas as pd
from typing import Union, List

class Chunker:
    """Update the chunk mixins
    """
    def chunk(documents: Union[pd.DataFrame, List], chunk_size: int=20):
        """
        Chunk an iterable object in Python.
        Args:
            documents:
                List of dictionaries/Pandas dataframe
            chunk_size:
                The chunk size of an object.
        Example:
            >>> documents = [{...}]
            >>> ViClient.chunk(documents)
        """
        if isinstance(documents, pd.DataFrame):
            for i in range(0, len(documents), chunk_size):
                yield documents.iloc[i : i + chunk_size]
        else:
            for i in range(0, len(documents), chunk_size):
                yield documents[i : i + chunk_size]
