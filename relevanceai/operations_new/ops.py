"""
This file is a mixin for dataset.
"""


class Operations:
    def label_from_dataset(self):
        raise NotImplementedError

    def label(
        self,
        label_documents,
        expanded=False,
        max_number_of_labels: int = 1,
        min_threshold: float = 0,
        similarity_metric: str = "cosine",
    ):
        """
        simple label of vectors against vectors

        Labelling performs a vector search on the labels and fetches the closest
        max_number_of_labels.

        Example
        --------

        .. code-block::

            ds = client.Dataset(...)
            # label an entire dataset
            ds.label(
                vector_field="sample_1_vector_",
                label_documents=[
                    {
                        "label": "value",
                        "price": 0.3,
                        "label_vector_": [1, 1, 1]
                    },
                    {
                        "label": "value-2",
                        "label_vector_": [2, 1, 1]
                    },
                ],
                expanded=True # stored as dict or list
            )
            # If missing "label", returns Error - labels missing `label` field
            # writes loop to set `label` field

            # If you want all values in a label document plus similarity, you need to set
            # expanded=True

        """
        from relevanceai.operations_new.label.ops import LabelOps

        ops = LabelOps()
        return ops.run()
