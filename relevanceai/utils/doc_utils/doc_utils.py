from collections.abc import MutableSequence

from typing import Dict, List, Any

from pandas import isna

from .chunk_doc_utils import ChunkDocUtils

try:
    from IPython.display import display
except ModuleNotFoundError:
    pass


class DocUtils(ChunkDocUtils):
    """Class for all document utilities.
    Primarily should be used as a mixin for future functions
    but can be a standalone.
    # TODO: Extend to Chunk Doc Reading and Chunk Doc Writing
    """

    pass


class Document(DocUtils):
    def __init__(self, document: Dict):
        """It takes a dictionary and returns a Document object

        Parameters
        ----------
        document : Dict
            Dict

        """
        super().__init__()

        for key, value in document.items():
            if isinstance(value, dict):
                document[key] = Document(value)

        self.data = document

    """> The `_ipython_display_` function is called by the IPython notebook when it wants to display an
    object

    """

    def _ipython_display_(self):
        display(self.json())

    def __repr__(self):
        """It returns a string representation of the object

        Returns
        -------
            The json() method is being called on the object.

        """
        return str(self.json())

    def __len__(self):
        """It returns the length of the JSON object

        Returns
        -------
            The length of the json object

        """
        return len(self.json())

    def __eq__(self, other):
        """If the keys and values of the two dictionaries are the same, then the dictionaries are the same

        Parameters
        ----------
        other
            The other object to compare to.

        """
        try:
            for key1, key2, value1, value2 in zip(
                self.keys(),
                other.keys(),
                self.values(),
                other.values(),
            ):
                if value1 != value2:
                    return False
            return True
        except:
            return False

    def __getattr__(self, name):
        """If the attribute name starts with two underscores, raise an AttributeError. Otherwise, if the
        attribute name is in the data dictionary, return the value from the dictionary. Otherwise, raise
        an AttributeError

        Parameters
        ----------
        name
            The name of the parameter.

        Returns
        -------
            A class object.

        """
        if name.startswith("__"):
            raise AttributeError
        if hasattr(self, "data"):
            if name in self.data:
                return self.data[name]
        raise AttributeError

    def __getitem__(self, key: str) -> Any:
        """It takes a string, splits it into a list of strings, and then uses the first string to get the
        first value, the second string to get the second value, and so on

        Parameters
        ----------
        key : str
            str

        Returns
        -------
            The value of the key.

        """
        levels = key.split(".")

        value = self.data
        for level in levels:
            value = value.__getitem__(level)

        return value

    """> The `__iter__` function returns an iterator object that iterates over the keys of the dictionary

    Returns
    -------
        The keys of the dictionary.

    """

    def __iter__(self):
        return iter(self.keys())

    def __setitem__(self, key: str, value: Any) -> None:
        """It takes a key and a value, splits the key into a list of keys, and then calls the setitem
        function with the list of keys and the value

        Parameters
        ----------
        key : str
            str
        value : Any
            The value to set.

        """
        self.setitem(self, key.split("."), value)

    def setitem(self, obj, keys: List[str], value: Any) -> None:
        """> It takes a list of keys, and a value, and sets the value at the end of the list of keys

        Parameters
        ----------
        obj
            The object to set the value on.
        keys : List[str]
            List[str]
        value : Any
            The value to set.

        """
        for key in keys[:-1]:
            obj = obj.data.setdefault(key, {})
        obj.data[keys[-1]] = value

    def json(self):
        """If the value is a DataFrame, then call the json() function on it. Otherwise, if the value is a
        NaN, then return None. Otherwise, return the value

        Returns
        -------
            A dictionary of dictionaries.

        """
        document = {}
        for key, value in self.data.items():
            if isinstance(value, self.__class__):
                document[key] = value.json()
            else:
                try:
                    if isna(value):
                        document[key] = None
                    else:
                        document[key] = value
                except:
                    document[key] = value
        return document

    def keys(self, parent=None):
        """It takes a dictionary, and returns a list of all the keys in the dictionary, including nested
        keys

        Parameters
        ----------
        parent
            The parent key of the current dictionary.

        Returns
        -------
            A list of keys

        """
        keys = {}

        for key, value in self.data.items():
            if isinstance(value, self.__class__):
                if parent is not None:
                    subparent = f"{parent}.{key}"
                else:
                    subparent = key
                keys.update({key: None for key in value.keys(parent=subparent)})
                keys.update({subparent: None})
            else:
                if parent is None:
                    keys[key] = None
                else:
                    keys[f"{parent}.{key}"] = None

        return keys.keys()

    def get(self, key, value=None):
        """If the key is in the dictionary, return the value, otherwise return the default value

        Parameters
        ----------
        key
            The key to be searched in the dictionary
        value
            The value to return if the key is not found. If not specified, None is returned.

        Returns
        -------
            The value of the key.

        """
        try:
            return self[key]
        except:
            return value

    def values(self):
        """It takes the dictionary, enumerates the keys, and then creates a new dictionary with the
        enumerated keys as the keys and the values as the values. Then it returns the values of the new
        dictionary

        Returns
        -------
            The values of the dictionary.

        """
        values = {i: self[key] for i, key in enumerate(self.keys())}
        return values.values()

    def items(self):
        """It returns a list of tuples, where each tuple is a key-value pair

        Returns
        -------
            A dictionary of the keys and values of the class.

        """
        items = {
            key: value for key, value in zip(list(self.keys()), list(self.values()))
        }
        return items.items()

    def update(self, other: Dict[str, Any]):
        """If the value is a dictionary, then create a new Document object with the value as the data.
        Otherwise, just set the value

        Parameters
        ----------
        other : Dict[str, Any]
            Dict[str, Any]

        """
        for key, value in other.items():
            if isinstance(value, dict):
                self.data[key] = Document(value)
            else:
                self.data[key] = value


class DocumentList(DocUtils, MutableSequence):
    """
    A Class for handling json like arrays of dictionaries

    Example:
    >>> docs = DocumentList([{"value": 2}, {"value": 10}])
    """

    def __init__(self, documents: List):
        """If the document is not an instance of Document, then create a new Document object with the
        document as the argument. Otherwise, just use the document as is

        Parameters
        ----------
        documents : List
            List

        """
        super().__init__()

        self.documents = [
            Document(document) if not isinstance(document, Document) else document
            for document in documents
        ]

    def _ipython_display_(self):
        """> The function takes a JSON object and displays it in a pretty format"""
        display(self.json())

    def __repr__(self):
        """It takes a list of documents and returns a list of dictionaries, where each dictionary
        represents a document

        Returns
        -------
            A list of dictionaries.

        """
        return repr([document.json() for document in self.documents])

    def __len__(self):
        """The function returns the length of the documents list

        Returns
        -------
            The length of the documents list.

        """
        return len(self.documents)

    def __add__(self, other):
        """The function takes two objects of the same class and adds the documents of the second object to
        the first object

        Parameters
        ----------
        other
            The other object to add to this one.

        Returns
        -------
            The documents are being returned.

        """
        self.documents += other.documents
        return self

    def __contains__(self, document):
        """The function returns True if the document is in the corpus, and False otherwise

        Parameters
        ----------
        document
            The document to check for.

        Returns
        -------
            The document is being returned.

        """
        return document in self.documents

    def __getitem__(self, index):
        """This function returns the document at the given index

        Parameters
        ----------
        index
            The index of the document to be retrieved.

        Returns
        -------
            The documents in the corpus.

        """
        return self.documents[index]

    def __setitem__(self, index, document):
        """It takes a document and adds it to the documents list

        Parameters
        ----------
        index
            The index of the document to be updated.
        document
            The document to be inserted.

        """
        if isinstance(document, dict):
            document = Document(document)
        assert isinstance(document, Document)
        self.documents[index] = document

    def __delitem__(self, index):
        """This function deletes the document at the given index

        Parameters
        ----------
        index
            The index of the item to be removed.

        """
        del self.documents[index]

    def insert(self, index, document):
        """It takes a document and inserts it into the list of documents at the specified index

        Parameters
        ----------
        index
            The index of the element before which to insert the new element.
        document
            The document to insert.

        """
        if isinstance(document, dict):
            document = Document(document)
        assert isinstance(document, Document)
        self.documents.insert(index, document)

    def json(self):
        """It returns a list of json objects.

        Returns
        -------
            A list of dictionaries.

        """
        return [document.json() for document in self.documents]
