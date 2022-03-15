from typing import Any, Optional, Union

import numpy as np
import pandas as pd

import plotly.express as px

from relevanceai.dataset.statistics.statistics import Statistics


class Plot(Statistics):
    def plot(
        self,
        x: Union[None, str] = None,
        y: Union[None, str] = None,
        z: Union[None, str] = None,
        dr_alias: Union[Any, str] = None,
        color: Optional[Union[None, str]] = None,
        number_of_documents: Union[None, int] = None,
        show_progress_bar: bool = True,
    ):
        euc_dims = [x, y, z]
        if [field is None for field in euc_dims].count(True) < 1 and dr_alias is None:
            raise ValueError

        if dr_alias is not None:
            if not dr_alias in self.schema:
                raise ValueError("Must reduce vectors before plotting")

        pd.options.plotting.backend = "plotly"

        select_fields = euc_dims + [dr_alias, color]
        select_fields = [field for field in select_fields if field is not None]

        print("Retrieving documents")
        if number_of_documents is None:
            documents = self.get_all_documents(
                select_fields=select_fields,
                show_progress_bar=show_progress_bar,
                include_vector=True,
            )
        else:
            documents = self.get_documents(
                number_of_documents=number_of_documents,
                select_fields=select_fields,
                include_vector=True,
            )

        if dr_alias is not None:
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
            df[color] = [sample[color] for sample in documents]

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
            df[color] = df[color].astype(str)

        if n_dims == 2:
            fig = px.scatter(df, **dims, color=color)

        elif n_dims == 3:
            fig = px.scatter_3d(df, **dims, color=color)

        fig.show()
