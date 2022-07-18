"""
Users can detect bias in a model by seeing which concepts certain vectors are closer to.
This is a particularly useful tool when users are looking at semantic vectors and
would like to check if certain words are leaning particularly towards any
specific category.

An example of analysing gender bias inside Google's Universal Sentence Encoder
can be found below.

.. code-block::

    # Set up the encoder
    !pip install -q sentence-transformers==2.2.0
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")

    from relevanceai.utils.bias_detection import bias_indicator
    bias_indicator(
        anchors=["boy", "girl"],
        values=["basketball", "draft", "skirt", "dress", "grave digger"],
        encoder=model.encode
    )

"""

from typing import List, Dict, Any, Callable

from relevanceai.utils import DocUtils
from relevanceai.utils.decorators.analytics import track_event_usage

from relevanceai.constants.errors import MissingPackageError


class BiasIndicator(DocUtils):
    def displacement_from_unit_line(self, v):
        """Distance from unit line"""
        proj = v[0] / 2 ** (1 / 2), v[1] / 2 ** (1 / 2)
        # Use the TanH activation function to prevent blow-up
        return (v[0] - proj[0]) ** 2 - (v[1] - proj[1]) ** 2

    def remove_box_line(self, axes, edge: str):
        axes[1].spines["top"].set_visible(False)

    def bias_indicator_to_html(
        self,
        anchor_documents: List[Dict],
        documents: List[Dict],
        metadata_field: str,
        vector_field: str,
        marker_colors=["purple", "#2E8B86"],
        white_sep_space: float = 0.4,
        xlabel="L2 Distance From Neutrality Line",
        title: str = None,
    ):
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:

            raise MissingPackageError("matplotlib")
        fig, axes = plt.subplots(ncols=2, sharey=True)
        anchor_values = self.get_field_across_documents(
            metadata_field, anchor_documents
        )
        graph_data = self.bias_indicator(
            anchor_documents=anchor_documents,
            documents=documents,
            metadata_field=metadata_field,
            vector_field=vector_field,
        )
        x1 = graph_data[
            self._get_bias_title(anchor_documents[0], metadata_field=metadata_field)
        ]
        x2 = graph_data[
            self._get_bias_title(anchor_documents[1], metadata_field=metadata_field)
        ]
        metadata = graph_data["metadata"]
        axes[0].barh(
            metadata, [abs(x) for x in x1], align="center", color=marker_colors[0]
        )
        axes[0].set(title=anchor_values[1])
        axes[1].barh(metadata, x2, align="center", color=marker_colors[1], zorder=10)
        axes[1].set(title=anchor_values[0])
        axes[0].invert_xaxis()
        axes[0].set(yticklabels=metadata)
        axes[0].yaxis.tick_right()
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["top"].set_visible(False)
        axes[0].spines["left"].set_visible(False)
        axes[0].spines["top"].set_visible(False)
        axes[0].set_xlabel(xlabel)
        axes[1].set_xticks(axes[0].get_xticks())
        fig.tight_layout()
        fig.subplots_adjust(wspace=white_sep_space)
        if title is None:
            fig.suptitle(
                f'Bias Indicator Between "{anchor_values[1]}" and "{anchor_values[0]}"',
                y=1.1,
            )
        else:
            fig.suptitle(title)

    def _get_bias_title(self, document, metadata_field):
        return f"bias_towards_{self.get_field(metadata_field, document)}"

    def bias_indicator(
        self,
        anchor_documents: List[Dict],
        documents: List[Dict],
        metadata_field: str,
        vector_field: str,
    ):
        """
        Bias Indicator returns a 0 if it is not the most similar group and a displacement value
        from neutrality towards the most similar group.
        """
        metadata = self.get_field_across_documents(metadata_field, documents)
        x = self.get_cosine_similarity_scores(
            documents, anchor_documents[0], vector_field
        )
        y = self.get_cosine_similarity_scores(
            documents, anchor_documents[1], vector_field
        )
        v = list(zip(x, y))
        displacement = [self.displacement_from_unit_line(_v) for _v in v]
        x1 = [min(_d, 0) for _d in displacement]
        x2 = [max(_d, 0) for _d in displacement]
        response = {}
        title = self._get_bias_title(anchor_documents[0], metadata_field)
        response[title] = x1
        title = self._get_bias_title(anchor_documents[1], metadata_field)
        response[title] = x2
        response["metadata"] = metadata
        return response

    def get_cosine_similarity_scores(
        self,
        documents: List[Dict[str, Any]],
        anchor_document: Dict[str, Any],
        vector_field: str,
    ) -> List[float]:
        """
        Compare scores based on cosine similarity

        Args:
            other_documents:
                List of documents (Python Dictionaries)
            anchor_document:
                Document to compare all the other documents with.
            vector_field:
                The field in the documents to compare
        Example:
            >>> documents = [{...}]
            >>> client.get_cosine_similarity_scores(documents[1:10], documents[0])
        """
        similarity_scores = []
        for i, doc in enumerate(documents):
            similarity_score = self.calculate_cosine_similarity(
                self.get_field(vector_field, doc),
                self.get_field(vector_field, anchor_document),
            )
            similarity_scores.append(similarity_score)
        return similarity_scores

    @staticmethod
    def calculate_cosine_similarity(a, b):
        from numpy import inner
        from numpy.linalg import norm
        from scipy import spatial

        return 1 - spatial.distance.cosine(a, b)
        # return inner(a, b) / (norm(a) * norm(b))


@track_event_usage("bias_indicator")
def bias_indicator(anchors: List, values: List, encoder: Callable):
    """
    Simple bias indicator based on vectors.

    .. code-block::

        from relevanceai.utils.bias_detection import bias_indicator
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-mpnet-base-v2")

        bias_indicator(
            anchors=["boy", "girl"],
            values=["basketball", "draft", "skirt", "dress", "grave digger"],
            encoder=model.encode
        )

    """
    # create the relevant documents
    anchor_docs = [{"value": a, "_vector_": encoder(a)} for a in anchors]
    value_docs = [{"value": a, "_vector_": encoder(a)} for a in values]
    return BiasIndicator().bias_indicator_to_html(
        anchor_documents=anchor_docs,
        documents=value_docs,
        metadata_field="value",
        vector_field="_vector_",
    )
