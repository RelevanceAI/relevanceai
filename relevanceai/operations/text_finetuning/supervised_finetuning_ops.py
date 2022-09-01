"""
.. warning::

    This is a beta feature and will be changing in the future. Do not use this in production systems.

Example
---------------------------------------

Train a text model using TripletLoss

You can find out more about different types of TripletLoss on https://www.sbert.net/docs/package_reference/losses.html

.. code-block::

    from relevanceai import Client
    client = Client()
    ds = client.Dataset("ecommerce")
    ops = SupervisedTripleLossFinetuneOps.from_dataset(
        dataset=ds,
        base_model="distilbert-base-uncased",
        chunksize=16,
        triple_loss_type:str='BatchHardSoftMarginTripletLoss'
    )
    ops.run(text_field="detail_desc", label_field="_cluster_.desc_use_vector_.kmeans-10", output_dir)

"""

import os
import json
import random
import copy
import logging
from typing import List, Optional, Any

try:
    from sentence_transformers import (
        SentenceTransformer,
        SentencesDataset,
        losses,
        evaluation,
    )
    from sentence_transformers.readers import InputExample
except ModuleNotFoundError:
    print(
        "Need to install SentenceTransformer by running `pip install sentence_transformers`."
    )

try:
    from torch.utils.data import DataLoader
except ModuleNotFoundError:
    print("Need to install SentenceTransformer by running `pip install torch-utils`.")

from relevanceai.operations.base import BaseOps
from relevanceai._api.api_client import APIClient
from relevanceai.client.helpers import Credentials


class SupervisedTripleLossFinetuneOps(APIClient, BaseOps):
    def __init__(
        self,
        dataset,
        base_model: str = "sentence-transformers/all-mpnet-base-v2",
        triple_loss_type: str = "BatchHardSoftMarginTripletLoss",
        chunksize: int = 32,
        save_best_model: bool = True,
        credentials: Optional[Credentials] = None,
    ):
        self.dataset = dataset
        self.base_model = base_model
        self.model = SentenceTransformer(base_model)
        self.chunksize = chunksize
        self.save_best_model = save_best_model
        self.loss_type = triple_loss_type
        self.evaluator = None

        if credentials is not None:
            super().__init__(credentials)

    def define_loss(self):
        if self.loss_type == "BatchAllTripletLoss":
            self.loss = losses.BatchAllTripletLoss(model=self.model)
        elif self.loss_type == "BatchHardSoftMarginTripletLoss":
            self.loss = losses.BatchHardSoftMarginTripletLoss(model=self.model)
        elif self.loss_type == "BatchHardTripletLoss":
            self.loss = losses.BatchHardTripletLoss(model=self.model)
        else:
            self.loss = losses.BatchSemiHardTripletLoss(model=self.model)

    def define_evaluator(
        self,
        text_data: List[str],
        labels: List[int],
        name="supervised_finetune_dev_eval",
    ):
        anchors, positives, negatives = self.build_triple_data(text_data, labels)
        self.evaluator = evaluation.TripletEvaluator(
            anchors=anchors, positives=positives, negatives=negatives, name=name
        )

    @staticmethod
    def build_triple_data(text_data, labels):
        # Only used for evaluation
        # anchors: Sentences to check similarity to
        # positives: List of positive sentences
        # negatives:  List of negative sentences

        n = 2  # ToDo: add checks, so that n can be an argument
        label_data_dict = {l: set() for l in set(labels)}
        for d, l in zip(text_data, labels):
            label_data_dict[l].add(d)
        label_data_dict = {l: list(label_data_dict[l]) for l in label_data_dict}
        all_cls = copy.deepcopy(label_data_dict)
        all_cls = list(all_cls.keys())
        for l in all_cls:
            if len(label_data_dict[l]) < 2 * n + 1:
                del label_data_dict[l]

        anchors = []
        positives = []
        negatives = []
        for l in label_data_dict:
            neg_list = list(label_data_dict.keys())
            neg_list.remove(l)
            anchors.extend(label_data_dict[l][:n])
            for idx in random.sample([j for j in range(n, len(label_data_dict[l]))], n):
                positives.append(label_data_dict[l][idx])
            for nl in random.sample(neg_list, n):
                idx = random.choice([j for j in range(n, len(label_data_dict[nl]))])
                negatives.append(label_data_dict[nl][idx])
        return anchors, positives, negatives

    def prepare_data_for_finetuning(self, text_data: List[str], labels: List[int]):
        data = [
            InputExample(texts=[text], label=label)
            for text, label in zip(text_data, labels)
        ]
        dataset = SentencesDataset(data, self.model)
        return DataLoader(dataset, shuffle=True, batch_size=self.chunksize)

    def fine_tune(
        self,
        train_data: List,
        dev_data: List = None,
        epochs: int = 3,
        output_path: str = "trained_model",
    ):
        if os.path.exists(output_path):
            print(
                "Output directory is detected. Assuming model was trained. Change output directory if you want to train a new model."
            )
        # Generates pairs for fine-tuning a model following the GPL algorithm. The final model is saved at  output_dir
        self.path_to_finetuned_model = output_path
        self.output_path = output_path
        print(f"Finetuning {self.base_model}")
        train_set, train_labels = train_data
        self.define_loss()
        dataloader = self.prepare_data_for_finetuning(train_set, train_labels)
        objectives = [(dataloader, self.loss)]
        if self.evaluator and dev_data:
            dev_set, dev_labels = dev_data
            self.define_evaluator(dev_set, dev_labels)
        self.model.fit(
            train_objectives=objectives,
            epochs=epochs,
            evaluator=self.evaluator,
            save_best_model=self.save_best_model,
            output_path=self.output_path,
        )
        print(f"Finished finetuning. Trained model is saved at {self.output_path}")

    def get_model(self, output_path: Optional[str] = None):
        if not self.output_path and not output_path:
            logging.warning("No Fine-Tuned Model Was Found")
        elif self.output_path:
            return SentenceTransformer(self.output_path)
        elif output_path:
            return SentenceTransformer(self.output_path)

    def fetch_text_and_labels_from_dataset(self, text_field, label_field):
        print("Fetching documents...")
        docs = self.dataset.get_all_documents(select_fields=[text_field, label_field])
        text_data = [doc[text_field] for doc in docs]
        labels = [self.client.get_field(label_field, doc) for doc in docs]
        labels_cls = set(labels)
        label_maps = {cl: i for i, cl in enumerate(labels_cls)}
        labels = [label_maps[l] for l in labels]
        return text_data, labels

    def run(
        self,
        text_field: str,
        label_field: str,
        epochs: int = 3,
        output_dir: str = "trained_model",
        percentage_for_dev: float = None,
    ):
        """
        Supervised finetuning a model using TripleLoss

        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()

            ds = client.Dataset("quickstart")
            from relevanceai.operations.text_finetuning.supervised_finetuning_ops import SupervisedTripleLossFinetuneOps
            ops = SupervisedTripleLossFinetuneOps.from_dataset(ds)
            ops.run(text_field="detail_desc", label_field="_cluster_.desc_use_vector_.kmeans-10", output_dir)

        Parameters
        -------------

        text_field: str
            The field you want to use as input text for fine-tuning
        label_field: str
            The field indicating the classes of the input
        output_dir: str
            The path of the output directory
        percentage_for_dev: float
            a number between 0 and 1 showing how much of the data should be used for evaluation. No evaluation if None

        """

        text_data, labels = self.fetch_text_and_labels_from_dataset(
            text_field, label_field
        )

        if percentage_for_dev:
            trin_size = int(len(labels) * (1 - percentage_for_dev))
            train_data = [text_data[:trin_size], labels[:trin_size]]
            dev_data = [text_data[trin_size:], labels[trin_size:]]
            self.do_evaluation = True
        else:
            train_data = [text_data, labels]
            dev_data = None
        self.fine_tune(train_data, dev_data, epochs, output_dir)

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
        **kwargs,
    ):
        cls = self(
            credentials=dataset.credentials,
            base_model=base_model,
            **kwargs,
        )
        cls.dataset_id = dataset.dataset_id
        return cls
