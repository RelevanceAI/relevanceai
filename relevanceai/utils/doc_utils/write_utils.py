from copy import deepcopy
from typing import List, Dict, Any
from .read_utils import DocReadUtils


class DocWriteUtils(DocReadUtils):
    """This is created as a Mixin for others to easily add to their classes"""

    def run_function_against_all_documents(self, fn, docs, field=None):
        """Run a function against documetns if the field is there"""
        # TODO: Change this to an operator and not a dict comprehension
        if field is not None:
            {fn(self.get_field(field, d)) for d in docs if self.is_field(field, d)}
        else:
            {fn(self.get_field(field, d)) for d in docs}

    @classmethod
    def _is_string_integer(cls, x):
        """Test if a string is numeric"""
        try:
            int(x)
            return True
        except:
            return False

    def set_fields_across_document(self, fields: List[str], doc: Dict, values: List):
        [self.set_field(f, doc, values[i]) for i, f in enumerate(fields)]

    @staticmethod
    def set_field(
        field: str, doc: Dict, value: Any, handle_if_missing=True, inplace=True
    ):
        """
        For nested dictionaries, tries to write to the respective field.
        If you toggle off handle_if_misisng, then it will output errors if the field is
        not found.
        e.g.
        field = kfc.item
        value = "fried wings"
        This should then create the following entries if they dont exist:
        {
            "kfc": {
                "item": "fried wings"
            }
        }
        Args:
            field:
                Field of the document to write.
            doc:
                Python dictionary
            value:
                Value to write
        Example:
            >>> from vectorai.client import ViClient
            >>> vi_client = ViClient(username, api_key, vectorai_url)
            >>> sample_document = {'kfc': {'item': ''}}
            >>> vi_client.set_field('kfc.item', sample_document, 'chickens')
        """
        if not inplace:
            doc = deepcopy(doc)

        fields = field.split(".")
        # Assign a pointer.
        d = doc
        for i, f in enumerate(fields):
            # Assign the value if this is the last entry e.g. stores.fastfood.kfc.item will be item
            if i == len(fields) - 1:
                d[f] = value
            else:
                if f in d.keys():
                    d = d[f]
                else:
                    d.update({f: {}})
                    d = d[f]

        if not inplace:
            return doc

    def set_field_across_documents(
        self, field: str, values: List[Any], docs: List[Dict]
    ):
        """
        For multiple documents, set the right fields.
        e.g.
        field = kfc.item
        value = "fried wings"
        This should then create the following entries if they dont exist:
        {
            "kfc": {
                "item": "fried wings"
            }
        }
        Args:
            field:
                Field of the document to write.
            doc:
                Python dictionary
            value:
                Value to write
        Example:
            >>> from vectorai.client import ViClient
            >>> vi_client = ViClient(username, api_key, vectorai_url)
            >>> sample_document = {'kfc': {'item': ''}}
            >>> vi_client.set_fields('kfc.item', sample_document, 'chickens')
        """
        assert len(values) == len(docs), (
            "Assert that the number of values " + "equates to the number of documents"
        )
        for i, value in enumerate(values):
            self.set_field(field, docs[i], value)
