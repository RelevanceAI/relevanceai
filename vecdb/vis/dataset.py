#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np

from pathlib import Path
from vecdb import VecDBClient
from vecdb_logging import create_logger
from dataclasses import dataclass, field
from typing import Union, Tuple, List


LOG = create_logger()


@dataclass(unsafe_hash=True)  # override base `__hash__`
class Dataset:
    user: User
    data_path: Path
    dataset_name: str
    max_vectors: Union[None, int]
    dataset_dir: Path = field(default_factory=Path)
    schema_path: Path = field(default_factory=Path)
    data: dict = field(default_factory=dict)
    schema: dict = field(default_factory=dict)

    def __post_init__(self):
        LOG.debug(f"Init VecDB client, {self.user.api_user, self.user.api_key}")
        self.vi = VecDBClient(self.user.api_user, self.user.api_key, base_url=URL)

        if not self.schema:
            LOG.debug(f"Loading schema")
            self.schema = self.load_schema(self.vi, self.schema_path, self.dataset_name)

        if not self.data:
            LOG.debug(f"Loading data")
            self.data = self.load_data(
                self.vi, self.dataset_dir, self.dataset_name, self.max_vectors
            )

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_data(
        vi: VecDBClient,
        dataset_dir: Path,
        dataset_name: str,
        max_vectors: Union[None, int],
    ) -> dict:
        # dataset_path = dataset_dir / f"{dataset_name}.json"
        dataset_path = dataset_dir / f"{dataset_name}.{max_vectors}.max.json"
        if not Path(dataset_path).exists():
            docs = vi.datasets.documents.list(dataset_name, page_size=1000)
            ## Saving subset for quick loading
            _sdataset_path = dataset_dir / f"_{dataset_name}.json"
            LOG.info(f"Downloading subset to {_sdataset_path}")
            ## Saving summary of vectors
            _sdata = docs["documents"][:10]
            for doc in _sdata:
                for k, v in doc.items():
                    if "_vector_" in k:
                        doc[k] = [np.min(v), np.mean(v), np.max(v)]
            json.dump(_sdata, open(_sdataset_path, "w"), ensure_ascii=False, indent=4)

            ## Paginating for full dataset
            LOG.info(f"Paginating: {dataset_name}")
            data = []
            _cursor = docs["cursor"]
            _page = 0
            while _cursor:
                docs = vi.datasets.documents.list(
                    dataset_name, page_size=1000, cursor=_cursor
                )
                _ddata = docs["documents"]
                _cursor = docs["cursor"]
                if (_ddata == []) or (_cursor is []):
                    break
                _pdataset_path = dataset_dir / f"{dataset_name}.{_page}.json"
                LOG.debug(f"Downloading to {_pdataset_path}")
                json.dump(
                    _ddata, open(_pdataset_path, "w"), ensure_ascii=False, indent=4
                )
                _page += 1
                data += _ddata
                if max_vectors and (len(data) >= int(max_vectors)):
                    break

            json.dump(data, open(dataset_path, "w"), ensure_ascii=False, indent=4)
            LOG.info(f"OUT_PATH: {dataset_path}")

        else:
            files = [
                f
                for f in Path(dataset_dir).glob("*.json")
                if f.name[0] != "_"
                if "max" not in f.name
            ]
            data = []
            for f in files:
                LOG.debug(f)
                if max_vectors and (len(data) >= int(max_vectors)):
                    break
                data += json.load(open(f))
            LOG.info(f"Loading from {dataset_path}")
            # data = json.load(open(dataset_path))
        return data

    @staticmethod
    def load_schema(vi: VecDBClient, schema_path: Path, dataset_name: str) -> dict:
        if not Path(schema_path).exists():
            schema = vi.datasets.schema(dataset_name)
            LOG.info(f"Downloading to {schema_path}")
            json.dump(schema, open(schema_path, "w"), ensure_ascii=False, indent=4)
        else:
            LOG.info(f"Loading from {schema_path}")
            schema = json.load(open(schema_path))
        return schema

    def valid_vector_name(self, vector_name: str) -> bool:
        if vector_name in self.schema.keys():
            if (type(self.schema[vector_name]) == dict) and self.schema[
                vector_name
            ].get("vector"):
                return True
            else:
                raise ValueError(f"{vector_name} is not a valid vector name")
        else:
            raise ValueError(f"{vector_name} is not in the {self.dataset_name} schema")

    def valid_label_name(self, label_name: str) -> bool:
        if label_name in self.schema.keys():
            if self.schema[label_name] in ["numeric", "text"]:
                return True
            else:
                raise ValueError(f"{label_name} is not a valid label name")
        else:
            raise ValueError(f"{label_name} is not in the {self.dataset_name} schema")
