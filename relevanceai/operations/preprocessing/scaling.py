from typing import Dict, List, Any, Union

from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)

import pandas as pd

from relevanceai.dataset.write.write import Write


class Scaler(Write):
    def scale(
        self,
        fields: List[Any],
        output_field=None,
        scaler: Any = None,
        number_of_documents: Union[int, None] = None,
        show_progress_bar: bool = True,
    ):

        filters = [
            {
                "field": field,
                "filter_type": "exists",
                "condition": "==",
                "condition_value": "",
            }
            for field in fields
        ]

        print("Getting documents")
        if number_of_documents is None:
            documents = self.get_all_documents(
                select_fields=fields,
                include_vector=True,
                filters=filters,
                show_progress_bar=show_progress_bar,
            )
        else:
            documents = self.get_documents(
                select_fields=fields,
                number_of_documents=number_of_documents,
                include_vector=True,
                filters=filters,
                show_progress_bar=show_progress_bar,
            )

        schema = self.schema
        vector_fields = [field for field in schema if "_vector_" in field]
        distributed_fields = []

        print("Scaling fields")
        data: Dict[str, Any] = {"_id": []}
        for document in documents:
            for field in document.keys():
                if field != "_id":
                    value = document[field]

                    if field in vector_fields:
                        if field not in distributed_fields:
                            distributed_fields.append(field)

                        n_dims = len(value)
                        for dim in range(n_dims):
                            vector_field_dim = f"{field}-{dim}"
                            if vector_field_dim not in data:
                                data[vector_field_dim] = []
                            data[vector_field_dim].append(value[dim])
                    else:
                        if field not in data:
                            data[field] = []
                        data[field].append(value)
                else:
                    data["_id"].append(document[field])

        df = pd.DataFrame(data)

        if scaler is None:
            scaler = RobustScaler()

        non_id_cols = [col for col in df.columns if col != "_id"]
        scaled = scaler.fit_transform(df[non_id_cols].values)
        for index, column in enumerate(non_id_cols):
            df[column] = scaled[:, index].tolist()

        documents = df.to_dict("records")

        for document in documents:
            for field in distributed_fields:
                dims = schema[field]["vector"]
                document[field if output_field is None else output_field] = [
                    document[f"{field}-{n}"] for n in range(dims)
                ]
                for n in range(dims):
                    document.pop(f"{field}-{n}")

        print("Updating fields")
        self.update_documents(
            self.dataset_id, documents, show_progress_bar=show_progress_bar
        )
