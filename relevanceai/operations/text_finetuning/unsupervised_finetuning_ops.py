"""
.. warning::

    This is a beta feature and will be changing in the future. Do not use this in production systems.

Example
---------------------------------------

Train a text model using GPL (Generative Pseudo-Labelling)
This can be helpful for `domain adaptation`.

You can find out more about GPL from: https://github.com/UKPLab/gpl

.. code-block::

    from relevanceai import Client
    client = Client()
    ds = client.Dataset("ecommerce")
    ops = GPLOps.from_dataset(dataset=ds,
        base_model="distilbert-base-uncased",
        t5_generator="BeIR/query-gen-msmarco-t5-base-v1",
        retrievers=["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
        cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunksize_gpl=16,
        output_path="trained_model",
    )
    ops.run(dataset=ds, text_field="detail_desc")

"""

import os
import json
import logging
from typing import List, Optional, Any

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
from relevanceai.client.helpers import Credentials


class GPLOps(APIClient, BaseOps):
    def __init__(
        self,
        base_model: str = "distilbert-base-uncased",
        t5_generator: str = "BeIR/query-gen-msmarco-t5-base-v1",
        retrievers: List[str] = ["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
        cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunksize_gpl: int = 16,
        output_path: str = "trained_model",
        credentials: Optional[Credentials] = None,
    ):
        self.base_model = base_model
        self.t5_generator = t5_generator
        self.retrievers = retrievers
        self.cross_encoder = cross_encoder
        self.chunksize_gpl = chunksize_gpl
        self.output_path = output_path
        if credentials is not None:
            super().__init__(credentials)

    def prepare_data_for_finetuning(
        self,
        documents: List[dict],
        text_field: str,
        dir_to_save_corpus: str,
        title_field: Optional[str] = None,
        corpus_filename: str = "corpus.jsonl",
    ):
        # Writes a corpus in the format compatible to the GPL library to later make Pos/Neg pairs
        corpus = []
        i = 0
        with open(f"{dir_to_save_corpus}/{corpus_filename}", "w") as jsonl:
            for doc in documents:
                if text_field not in doc:
                    continue
                line = {
                    "_id": str(i),
                    "text": doc.get(text_field, "").replace("\n", " "),
                    "metadata": {
                        f: doc[f] for f in doc if f not in [text_field, title_field]
                    },
                }
                if title_field is not None:
                    line["title"] = doc.get(title_field, "")
                else:
                    line["title"] = ""
                # iteratively write lines to the JSON lines corpus.jsonl file
                jsonl.write(json.dumps(line) + "\n")
                i += 1
                corpus.append(line)
        logging.info(
            f"A corpus of {len(corpus)} documents is saved at {dir_to_save_corpus}/corpus.jsonl"
        )

    def fine_tune(
        self,
        path_to_generated_data: str = ".",
        output_dir: str = "trained_model",
        gpl_steps: int = 500,
        do_evaluation: bool = False,
        qgen_prefix: str = "qgen",
        **gpl_kwargs,
    ):
        if os.path.exists(output_dir):
            print(
                "Output directory is detected. Assuming model was trained. Change output directory if you want to train a new model."
            )
        # Generates pairs for fine-tuning a model following the GPL algorithm. The final model is saved at  output_dir
        self.path_to_finetuned_model = output_dir
        self.output_path = output_dir
        # TODO: Add the rest of the parameters :(
        gpl.train(
            path_to_generated_data=path_to_generated_data,
            base_ckpt=self.base_model,
            batch_size_gpl=self.chunksize_gpl,
            gpl_steps=gpl_steps,
            output_dir=output_dir,
            generator=self.t5_generator,
            retrievers=self.retrievers,
            cross_encoder=self.cross_encoder,
            qgen_prefix=qgen_prefix,
            do_evaluation=do_evaluation,
            **gpl_kwargs,
        )

    def get_model(self, output_path: Optional[str] = None):
        if not self.output_path and not output_path:
            logging.warning("No Fine-Tuned Model Was Found")
        elif self.output_path:
            return SentenceTransformer(self.output_path)
        elif output_path:
            return SentenceTransformer(self.output_path)

    def run(
        self,
        dataset: str,
        text_field: str,
        gpl_steps: int = 500,
        path_to_generated_data: str = ".",
        output_dir: str = "trained_model",
        dir_to_save_corpus: str = ".",
        do_evaluation: bool = False,
    ):
        """
        Finetune a model using Generative Pseudo-Labelling

        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()

            ds = client.Dataset("quickstart")
            # !pip install -q gpl
            from relevanceai.operations.text_finetuning.unsupervised_finetuning_ops import GPLOps
            ops = GPLOps.from_dataset(ds)
            ops.run(dataset=ds, text_field="product_title")

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
        docs = self._get_all_documents(
            dataset_id=self._get_dataset_id(dataset), select_fields=[text_field]
        )

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

    @classmethod
    def from_client(self, client, *args, **kwargs):
        credentials = client.credentials
        return self(
            credentials=credentials,
            *args,
            **kwargs,
        )

    @classmethod
    def from_dataset(
        self,
        dataset: Any,
        base_model: str = "distilbert-base-uncased",
        t5_generator: str = "BeIR/query-gen-msmarco-t5-base-v1",
        retrievers: List[str] = ["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
        cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunksize_gpl: int = 16,
        output_path: str = "trained_model",
        **kwargs,
    ):
        cls = self(
            credentials=dataset.credentials,
            base_model=base_model,
            t5_generator=t5_generator,
            retrievers=retrievers,
            cross_encoder=cross_encoder,
            chunksize_gpl=chunksize_gpl,
            output_path=output_path,
            **kwargs,
        )
        cls.dataset_id = dataset.dataset_id
        return cls
