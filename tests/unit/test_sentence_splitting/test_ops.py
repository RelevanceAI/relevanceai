from relevanceai.dataset import Dataset

from relevanceai import mock_documents

from relevanceai.operations_new.processing.text.sentence_splitting.ops import (
    SentenceSplitterOps,
)


class TestSentenceSplitterOps:
    def test_split_text(self):
        ops = SentenceSplitterOps()
        text = """Loudermilk and GOP Rep. Rodney Davis, the top Republican on the House Administration Committee,
        issued a joint statement later Thursday responding to the committee's letter again pushing back on
        any allegation of "reconnaissance" tours on January 5 and calling for Capitol Police to release
        the footage."""
        texts = ops.split_text(
            text=text,
            language="en",
        )
        for t in texts:
            assert t in text

    def test_split_text_document(self):
        ops = SentenceSplitterOps()
        document = {
            "status": 0,
            "params": {
                "facet_range": "initial_release_date",
                "facet_limit": "300",
                "facet_range_gap": "+1YEAR",
            },
        }
        documents = ops.split_text_document(
            text_field="params.facet_limit", document=document, output_field="output"
        )
        assert documents["output"][0] == {"params.facet_limit": "300"}
