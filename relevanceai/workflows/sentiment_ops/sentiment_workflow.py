from relevanceai.workflows.sentiment_ops.sentiments import SentimentOps
from relevanceai.workflows.base import Workflow


class SentimentWorkflow(Workflow, SentimentOps):
    def __init__(
        self,
        workflow_alias: str = "sentiment-analysis",
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
    ):
        """
        workflow
        """
        self.workflow_alias = workflow_alias
        self.model_name = model_name

    def fit_dataset(
        self,
        dataset,
        input_field: str,
        output_field: str,
        log_to_file: bool = True,
        chunksize: int = 20,
        workflow_alias: str = "sentiment",
        notes=None,
    ):
        workflow = Workflow(
            self.analyze_sentiment, workflow_alias=workflow_alias, notes=notes
        )
        workflow.fit_dataset(
            dataset=dataset,
            input_field=input_field,
            output_field=output_field,
            log_to_file=log_to_file,
            chunksize=chunksize,
        )
