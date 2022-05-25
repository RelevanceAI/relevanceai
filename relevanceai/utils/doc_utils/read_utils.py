import pandas as pd
from typing import Dict, List, Any
from .errors import MissingFieldError


class DocReadUtils:
    """This is created as a Mixin for others to easily add to their classes"""

    @classmethod
    def get_field(self, field: str, doc: Dict, missing_treatment="raise_error"):
        """
        For nested dictionaries, tries to access a field.
        e.g.
        field = kfc.item
        This should return "chickens" based on doc below.
        {
            "kfc": {
                "item": "chickens"
            }
        }
        Args:
            field:
                Field of a document.
            doc:
                document
            missing_treatment:
                Can be one of return_empty_string/return_none/raise_error
        Example:
            >>> from vectorai.client import ViClient
            >>> vi_client = ViClient(username, api_key, vectorai_url)
            >>> sample_document = {'kfc': {'item': 'chicken'}}
            >>> vi_client.get_field('kfc.item', sample_document) == 'chickens'
        """
        d = doc
        for f in field.split("."):
            try:
                d = d[f]
            except KeyError:
                try:
                    return doc[field]
                except KeyError:
                    if missing_treatment == "return_none":
                        return None
                    elif missing_treatment == "return_empty_string":
                        return ""
                    elif missing_treatment == "raise_error":
                        raise MissingFieldError(
                            "Document is missing " + f + " of " + field
                        )
                    else:
                        return missing_treatment
            except TypeError:
                if self._is_string_integer(f):
                    # Get the Get the chunk document out.
                    try:
                        d = d[int(f)]
                    except IndexError:
                        pass
                else:
                    if missing_treatment == "return_none":
                        return None
                    elif missing_treatment == "return_empty_string":
                        return ""
                    raise MissingFieldError("Document is missing " + f + " of " + field)
        return d

    @classmethod
    def _is_string_integer(cls, x):
        """Test if a string is numeric"""
        try:
            int(x)
            return True
        except:
            return False

    @classmethod
    def get_fields(
        self, fields: List[str], doc: Dict, missing_treatment="return_empty_string"
    ) -> List[Any]:
        """
        For nested dictionaries, tries to access a field.
        e.g.
        field = kfc.item
        This should return "chickens" based on doc below.
        {
            "kfc": {
                "item": "chickens"
            }
        }
        Args:
            fields:
                List of fields of a document.
            doc:
                document
        Example:
            >>> from vectorai.client import ViClient
            >>> vi_client = ViClient(username, api_key, vectorai_url)
            >>> sample_document = {'kfc': {'item': 'chicken'}}
            >>> vi_client.get_field('kfc.item', sample_document) == 'chickens'
        """
        return [self.get_field(f, doc, missing_treatment) for f in fields]

    def get_field_across_documents(
        self,
        field: str,
        docs: List[Dict],
        missing_treatment: str = "return_empty_string",
    ):
        """
        For nested dictionaries, tries to access a field.
        e.g.
        field = kfc.item
        This should return "chickens" based on doc below.
        {
            "kfc": {
                "item": "chickens"
            }
        }
        Args:
            fields:
                List of fields of a document.
            doc:
                document
            missing_treatment:
                This can be one of 'skip', 'return_empty_string'

        Example:
            >>> from vectorai.client import ViClient
            >>> vi_client = ViClient(username, api_key, vectorai_url)
            >>> documents = vi_client.create_sample_documents(10)
            >>> vi_client.get_field_across_documents('size.cm', documents)
            # returns 10 values in the nested dictionary
        """
        if missing_treatment == "skip":
            # Returns only the relevant documents
            def check_field_in_doc(doc):
                return self.is_field(field, doc)

            return [
                self.get_field(field, doc) for doc in filter(check_field_in_doc, docs)
            ]
        return [self.get_field(field, doc, missing_treatment) for doc in docs]

    def get_fields_across_document(
        self, fields: List[str], doc: Dict, missing_treatment="return_empty_string"
    ):
        """
        Get numerous fields across a document.
        """
        return [
            self.get_field(f, doc, missing_treatment=missing_treatment) for f in fields
        ]

    def get_fields_across_documents(
        self,
        fields: List[str],
        docs: List[Dict],
        missing_treatment="return_empty_string",
    ):
        """Get numerous fields across documents.

        Example
        ----------

        For document:

        docs = [
            {
                "value": 2,
                "type": "car"
            },
            {
                "value": 10,
                "type": "bus"
            }
        ]

        >>> DocUtils().get_fields_across_documents(["value", "type"], docs)
        >>> [2, "car", 10, "bus"]

        Parameters
        --------------

        missing_treatment: str
            Can be one of ["skip", "return_empty_string", "return_none", "skip_if_any_missing"]
            If "skip_if_any_missing", the document will not be included if any field is missing

        """
        if missing_treatment == "skip_if_any_missing":
            docs = self.filter_docs_for_fields(fields, docs)
            return [self.get_fields_across_document(fields, doc) for doc in docs]
        return [
            self.get_fields_across_document(
                fields, doc, missing_treatment=missing_treatment
            )
            for doc in docs
        ]

    def filter_docs_for_fields(self, fields: List, docs: List):
        """
        Filter for docs if they contain a list of fields
        """

        def is_any_field_missing_in_doc(doc):
            return all([self.is_field(f, doc) for f in fields])

        return filter(is_any_field_missing_in_doc, docs)

    @classmethod
    def is_field(self, field: str, doc: Dict) -> bool:
        """
        For nested dictionaries, tries to access a field.
        e.g.
        field = kfc.item
        This should return "chickens" based on doc below.
        {
            "kfc": {
                "item": "chickens"
            }
        }
        Args:
            collection_name:
                Name of collection.
            job_id:
                ID of the job.
            job_name:
                Name of the job.
        Example:
            >>> from vectorai.client import ViClient
            >>> vi_client = ViClient(username, api_key, vectorai_url)
            >>> sample_document = {'kfc': {'item': 'chicken'}}
            >>> vi_client.is_field('kfc.item', sample_document) == True
        """
        d = doc
        for f in field.split("."):
            try:
                d = d[f]
            except KeyError:
                try:
                    doc[field]
                    return True
                except KeyError:
                    try:
                        d[field]
                    except KeyError:
                        return False
            except TypeError:
                # To Support integers
                if self._is_string_integer(f):
                    # Get the Get the chunk document out.
                    try:
                        d = d[int(f)]
                    except IndexError:
                        pass
                else:
                    return False
        return True

    def is_field_across_documents(self, field, documents):
        return all([self.is_field(field, doc)] for doc in documents)

    @staticmethod
    def list_doc_fields(doc: dict) -> List[str]:
        """returns all fields in a document, nested fields are flattened
        example:
        input: doc = {'a': {'b':'v', 'c':'v'},
                      'd':'v'}
                      'e':{'f':{'g':'v'}
        output: ['d', 'a.b', 'a.c', 'e.f.g']
        """
        df = pd.json_normalize(doc, sep=".")
        return list(df.columns)

    @classmethod
    def subset_documents(
        self,
        fields: List[str],
        docs: List[Dict],
        missing_treatment: str = "return_none",
    ) -> List[Dict]:
        """
        Args:
            fields:
                A list of fields of interest.
            docs:
                A list of documents that may or may not have the chosen
                fields.
            missing_treatment:
                Cane be on of return_empty_string/return_none/raise_error
        Example:
            >>> from vectorai.client import ViClient
            >>> vi_client = ViClient(username, api_key, vectorai_url)
            >>> docs = [
            ...     {"kfc": {"food": "chicken nuggets", "drink": "soda"}}
            ...     {"mcd": {"food": "hamburger", "drink": "pop"}}
            ... ]
            >>> fields = [
            ...     "kfc.food", "kfc.drink", "mcd.food", "mcd.drink"
            ... ]
            >>> vi_client.subset_documents(fields, docs) == [
            ...     {
            ...         "kfc.food": "chicken nuggets", "kfc.drink": "soda"},
            ...         "mcd.food": "", "mcd.drink": ""
            ...     },
            ...     {
            ...         "kfc.food": "", "kfc.drink": ""},
            ...         "mcd.food": "hamburger", "mcd.drink": "pop"},
            ...     }
            ... ]
        """
        return [
            {
                field: self.get_field(field, doc, missing_treatment=missing_treatment)
                for field in fields
            }
            for doc in docs
        ]
