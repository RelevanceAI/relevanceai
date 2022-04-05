from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from relevanceai.dataset.read.statistics import Statistics


class Plot(Statistics):
    def scatter(
        self,
        x: Optional[Any] = None,
        y: Optional[Any] = None,
        z: Optional[Any] = None,
        vector_field: Optional[Any] = None,
        alias: Optional[Any] = None,
        color: Optional[Any] = None,
        height: int = 800,
        width: int = 1200,
        number_of_documents: Union[None, int] = None,
        show_progress_bar: bool = True,
    ):
        try:
            import plotly.express as px
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "plotly package not found. "
                "Please install plotly with `python -m "
                "pip install -U plotly` to install "
                "plotly."
            )

        if vector_field and alias:
            dr_alias = f"_dr_.{alias}.{vector_field}"
        else:
            dr_alias = ""

        euc_dims = [x, y, z]
        if [not field is None for field in euc_dims].count(True) < 1 and dr_alias == "":
            raise ValueError

        if dr_alias:
            if not dr_alias in self.schema:
                raise ValueError("Must reduce vectors before plotting")

        if f"_cluster_.{vector_field}.{color}" in self.schema:
            shorthand = color
            color = f"_cluster_.{vector_field}.{color}"

        select_fields = euc_dims + [dr_alias, color]
        select_fields = [
            field for field in select_fields if field is not None and field
        ]

        print("Retrieving documents")

        filters = [
            {
                "field": field,
                "filter_type": "exists",
                "condition": "==",
                "condition_value": "",
            }
            for field in select_fields
        ]
        if number_of_documents is None:
            documents = self.get_all_documents(
                select_fields=select_fields,
                show_progress_bar=show_progress_bar,
                include_vector=True,
                filters=filters,
            )
        else:
            documents = self.get_documents(
                number_of_documents=number_of_documents,
                select_fields=select_fields,
                include_vector=True,
                filters=filters,
            )

        if color is not None:
            if "_cluster_" in color:
                for document in documents:
                    try:
                        document[color] = list(
                            list(document["_cluster_"].values())[0].values()
                        )[0]
                        document.pop("_cluster_")
                    except:
                        pass

        n_dims: Union[int, str]
        if dr_alias:
            dr_algo, n_dims = dr_alias.split(".")[1].split("-")
            n_dims = int(n_dims)

            vectors = np.array([sample[dr_alias] for sample in documents])
            columns = [f"{dr_algo}{index}" for index in range(n_dims)]
            dims = {
                dim: f"{dr_algo}{index}"
                for dim, index in zip(["x", "y", "z"][:n_dims], range(n_dims))
            }
            df = pd.DataFrame(vectors, columns=columns)
            df["_id"] = [sample["_id"] for sample in documents]

        else:
            df = pd.DataFrame(documents)
            dims = {}
            if x is not None:
                dims["x"] = x
            if y is not None:
                dims["y"] = y
            if z is not None:
                dims["z"] = z
            n_dims = len(dims)

        if color is not None:
            if "shorthand" in locals():
                df[shorthand] = [sample[color] for sample in documents]
                color = shorthand
            else:
                df[color] = [sample[color] for sample in documents]

            df[color] = df[color].astype(str)

        if n_dims == 2:
            fig = px.scatter(df, **dims, color=color, width=width, height=height)

        elif n_dims == 3:
            fig = px.scatter_3d(df, **dims, color=color, width=width, height=height)

        fig.show()

    def show_workflows(self):
        from relevanceai.recipes.diagrams import create_diagram

        workflows = self.metadata["workflows"]
        return create_diagram(workflows)
