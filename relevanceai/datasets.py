import requests
from relevanceai import config
from relevanceai.auth import Auth
from relevanceai._request import handle_response
from relevanceai.steps.vector_search import VectorSimilaritySearch
from typing import Any, Dict, List, Optional, Union


class Dataset:
    def __init__(
        self,
        id: str,
        auth: Auth = None,
    ):
        self.id = id
        self.auth: Auth = config.auth if auth is None else auth
        try:
            from vecdb.collections.dataset import Dataset
            from vecdb.api.local import Client

            self.vecdb_client = Client(
                f"{self.auth.project}:{self.auth.api_key}:{self.auth.region}",
                authenticate=False,
            )
        except ImportError:
            raise ImportError(
                "vecdb is not installed. Please install vecdb with `pip install vecdb`"
            )
        self.db = Dataset(api=self.vecdb_client.api, dataset_id=self.id)
    

    def insert(
        self,
        documents: List = None,
        ids: List[str] = None,
        data: List[str] = None,
        metadata: List[Dict[str, Any]] = None,
        vector: List[List[float]] = None,
        encoders: List = None,
        *args,
        **kwargs,
    ):
        if len(documents) > 100:
            return self.db.bulk_insert(
                documents=documents,
                ids=ids,
                data=data,
                metadata=metadata,
                vector=vector,
                encoders=encoders,
                *args,
                **kwargs,
            )
        else:
            return self.db.insert(
                documents=documents,
                ids=ids,
                data=data,
                metadata=metadata,
                vector=vector,
                encoders=encoders,
                *args,
                **kwargs,
            )


    def search(
        self,
        text: str,
        field: str = "text_vector_",
        page_size: int = 5,
        model: str = "all-mpnet-base-v2",
        return_as_step: bool = False,
    ):
        if "_vector_" not in field:
            field = f"{field}_vector_"
        if return_as_step:
            return VectorSimilaritySearch(
                dataset_id=self.id,
                query=text,
                vector_field=field,
                model=model,
                page_size=page_size
            )
        else:
            return VectorSimilaritySearch(
                dataset_id=self.id,
                query=text,
                vector_field=field,
                model=model,
                page_size=page_size
            ).run()


    def delete(self):
        return self.db.delete()


    def all(self):
        return self.db.get_all()


def list_datasets(auth: Auth = None):
    auth: Auth = config.auth if auth is None else auth
    try:
        from vecdb.collections.dataset import Dataset
        from vecdb.api.local import Client

        vecdb_client = Client(
            f"{auth.project}:{auth.api_key}:{auth.region}",
            authenticate=False,
        )
    except ImportError:
        raise ImportError(
            "vecdb is not installed. Please install vecdb with `pip install vecdb`"
        )
    return vecdb_client.list_datasets()
