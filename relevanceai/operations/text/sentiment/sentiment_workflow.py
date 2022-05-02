from relevanceai.operations.text.sentiment.sentiments import SentimentOps
from relevanceai.workflow.base import Workflow
from typing import Optional


class SentimentWorkflow(Workflow, SentimentOps):
    """
    Sentiment workflow
    """

    def __init__(
        self, model_name, workflow_alias: str = "sentiment", **workflow_kwargs
    ):
        self.model_name = model_name
        super().__init__(
            func=self.analyze_sentiment,
            workflow_alias=workflow_alias,
            **workflow_kwargs
        )

    def fit_dataset(
        self,
        dataset,
        input_field: str,
        output_field: str = "_sentiment_",
        log_to_file: bool = True,
        chunksize: int = 20,
        workflow_alias: str = "sentiment",
        notes=None,
        refresh: bool = False,
        highlight: bool = False,
        positive_sentiment_name: str = "positive",
        max_number_of_shap_documents: Optional[int] = None,
        min_abs_score: float = 0.1,
    ):
        def analyze_sentiment(text):
            return self.analyze_sentiment(
                text,
                highlight=highlight,
                positive_sentiment_name=positive_sentiment_name,
                max_number_of_shap_documents=max_number_of_shap_documents,
                min_abs_score=min_abs_score,
            )

        workflow = Workflow(
            analyze_sentiment, workflow_alias=workflow_alias, notes=notes
        )
        return workflow.fit_dataset(
            dataset=dataset,
            input_field=input_field,
            output_field=output_field,
            log_to_file=log_to_file,
            chunksize=chunksize,
            refresh=refresh,
        )
