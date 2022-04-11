"""
GPL
"""

import os
import json
import logging
from typing import List, Optional

try:
    import gpl
except ModuleNotFoundError:
    print("Need to install gpl by running `pip install gpl`.")

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    print(
        "Need to install SentenceTransformer by running `pip install sentence_transformers`."
    )
from relevanceai.operations.base import BaseOps
from relevanceai._api.api_client import APIClient


class GPLOps(BaseOps, APIClient):
    def __init__(
        self,
        base_model: str,
        t5_generator: str = "BeIR/query-gen-msmarco-t5-base-v1",
        retrievers: List[str] = ["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
        cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size_gpl: int = 16,
    ):
        self.base_model = base_model
        self.t5_generator = t5_generator
        self.retrievers = retrievers
        self.cross_encoder = cross_encoder
        self.batch_size_gpl = batch_size_gpl
        self.output_path = None

    def prepare_data_for_finetuning(
        self,
        documents: List[dict],
        text_field: str,
        dir_to_save_corpus: str,
        title_field: str = None,
    ):
        # Writes a corpus in the format compatible to the GPL library to later make Pos/Neg pairs
        corpus = []
        i = 0
        with open(f"{dir_to_save_corpus}/corpus.jsonl", "w") as jsonl:
            for doc in documents:
                if text_field not in doc:
                    continue
                line = {
                    "_id": str(i),
                    "title": doc[title_field].replace("\n", " ")
                    if title_field in doc
                    else "",
                    "text": doc[text_field].replace("\n", " "),
                    "metadata": {
                        f: doc[f] for f in doc if f not in [text_field, title_field]
                    },
                }
                # iteratively write lines to the JSON lines corpus.jsonl file
                jsonl.write(json.dumps(line) + "\n")
                i += 1
                corpus.append(line)
        logging.info(
            f"A corpus of {len(corpus)} documents is saved at {dir_to_save_corpus}/corpus.jsonl"
        )

    def fine_tune(
        self,
        path_to_generated_data: str,
        output_dir: str = "trained_model",
        gpl_steps: int = 500,
        do_evaluation: bool = False,
    ):
        if os.path.exists(output_dir):
            print(
                "Output directory is detected. Assuming model was trained. Change output directory if you want to train a new model."
            )
        # Generates pairs for fine-tuning a model following the GPL algorithm. The final model is saved at  output_dir
        self.path_to_finetuned_model = output_dir
        self.output_path = output_dir
        gpl.train(  # type: ignore
            path_to_generated_data=path_to_generated_data,
            base_ckpt=self.base_model,
            batch_size_gpl=self.batch_size_gpl,
            gpl_steps=gpl_steps,
            output_dir=output_dir,
            generator=self.t5_generator,
            retrievers=self.retrievers,
            cross_encoder=self.cross_encoder,
            qgen_prefix="qgen",
            do_evaluation=do_evaluation,
        )

    def get_finetuned_model(self, output_path: Optional[str] = None):
        if not self.output_path and not output_path:
            logging.warning("No Fine-Tuned Model Was Found")
        elif self.output_path:
            return SentenceTransformer(self.output_path)
        elif output_path:
            return SentenceTransformer(self.output_path)

    def operate(
        self,
        dataset: str,
        text_field: str,
        gpl_steps: int = 500,
        path_to_generated_data: str = "temp_data",
        output_dir: str = "trained_model",
        dir_to_save_corpus: str = ".",
        do_evaluation: bool = False,
    ):
        """
        Finetune a model

        Example
        -----------

        # TODO

        Parameters
        -------------
        text_field: str
            The field you want to use for fine-tuning
        dir_to_save_corpus: str
            path to save the corpus that is going to be used by the GPL alg.
        gpl_steps: int
            The number of steps in Generative Pseudo Labelling
        path_to_generated_data: str
            The path to generated data
        output_dir: str
            The path of the output directory
        dir_to_save_corpus: str
            The directory to save corpus
        do_evaluation: bool
            If True, it performs the evaluation

        """
        print("Fetching documents...")
        docs = self._get_all_documents(dataset_id=dataset)

        print("Preparing training data...")
        self.prepare_data_for_finetuning(
            documents=docs,
            text_field=text_field,
            dir_to_save_corpus=dir_to_save_corpus,
        )

        print("Training Model...")
        self.fine_tune(
            path_to_generated_data=path_to_generated_data,
            output_dir=output_dir,
            gpl_steps=gpl_steps,
            do_evaluation=do_evaluation,
        )

        print(f"Trained model saved at {self.output_path}")
