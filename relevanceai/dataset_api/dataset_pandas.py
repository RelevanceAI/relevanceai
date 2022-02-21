import pandas as pd

from abc import ABC


class PandasSeries(ABC):
    def __init__(self):
        raise Exception("Cannot be instantiated")

    def _get_pandas_series(self):
        documents = self._get_documents(
            dataset_id=self.dataset_id, select_fields=[self.field], include_vector=False
        )

        try:
            df = pd.DataFrame(documents)
            df.set_index("_id", inplace=True)
            return df.squeeze()
        except KeyError:
            raise Exception("No documents found")

    @property
    def T(self) -> pd.Series:
        series = self._get_pandas_series()
        return series.T

    def tolist(self) -> list:
        series = self._get_pandas_series()
        return series.tolist()

    to_list = tolist


class PandasDataFrame(ABC):
    def __init__(self):
        raise Exception("Cannot be instantiated")
