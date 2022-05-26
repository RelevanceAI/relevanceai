"""Utilities for Chunk Documents
"""
import itertools
from typing import Callable
from .write_utils import DocWriteUtils


class ChunkDocUtils(DocWriteUtils):
    @classmethod
    def get_chunk(cls, chunk_field, doc):
        return cls.access_document_field(chunk_field, doc)

    @classmethod
    def get_field_across_chunks(cls, chunk_field, field, doc):
        chunk = cls.get_chunk(chunk_field, doc)
        return [cls.get_field(field, d) for d in chunk]

    def run_function_across_chunks(
        self,
        function: Callable,
        chunk_field,
        field: str = None,
        output_field: str = None,
        doc={},
    ):
        """Run a function on a field across chunks
        Params:
        function:
            Any function to run across documents
        field:
            The field is AFTER the Chunk Field
        """
        if field is None:
            chunk_values = self.get_chunk(chunk_field, doc)
        else:
            chunk_values = self.get_field_across_chunks(chunk_field, field, doc)
        if output_field is None:
            return map(function, chunk_values)
        self.set_field_across_documents(
            output_field, list(map(function, chunk_values)), chunk_values
        )

    def run_function_across_chunks_across_docs(
        self,
        function: Callable,
        chunk_field: str,
        field: str = None,
        output_field: str = None,
        docs=[],
    ):
        """Run a function across chunk field in documents"""

        def map_run_function_across_chunks(doc):
            return self.run_function_across_chunks(
                function, chunk_field, field, output_field, doc
            )

        return map(map_run_function_across_chunks, docs)
