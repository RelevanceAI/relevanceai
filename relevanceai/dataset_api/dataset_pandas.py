import numpy as np
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

    # Attributes

    @property
    def index(self) -> pd.core.indexes.base.Index:
        series = self._get_pandas_series()
        return series.index

    @property
    def array(self) -> pd.core.arrays.numpy_.PandasArray:
        series = self._get_pandas_series()
        return series.array

    @property
    def values(self) -> np.ndarray:
        series = self._get_pandas_series()
        return series.values

    @property
    def dtype(self) -> np.dtype:
        series = self._get_pandas_series()
        return series.dtype

    @property
    def shape(self) -> tuple:
        series = self._get_pandas_series()
        return series.shape

    @property
    def nbytes(self) -> int:
        series = self._get_pandas_series()
        return series.nbytes

    @property
    def ndim(self) -> int:
        series = self._get_pandas_series()
        return series.ndim

    @property
    def size(self) -> int:
        series = self._get_pandas_series()
        return series.size

    @property
    def T(self) -> pd.Series:
        series = self._get_pandas_series()
        return series.T

    def memory_usage(self, index=True, deep=False) -> int:
        series = self._get_pandas_series()
        return series.memory_usage(index=index, deep=deep)

    @property
    def hasnans(self) -> bool:
        series = self._get_pandas_series()
        return series.hasnans

    @property
    def empty(self) -> bool:
        series = self._get_pandas_series()
        return series.empty

    @property
    def dtypes(self) -> bool:
        series = self._get_pandas_series()
        return series.dtypes

    @property
    def name(self) -> str:
        series = self._get_pandas_series()
        return series.name

    @property
    def flags(self) -> pd.core.flags.Flags:
        series = self._get_pandas_series()
        return series.flags

    def set_flags(self, *, copy=False, allows_duplicate_labels=None) -> pd.Series:
        series = self._get_pandas_series()
        return series.set_flags(
            copy=copy, allows_duplicate_labels=allows_duplicate_labels
        )

    # Conversion

    def astype(self, dtype, copy=True, errors="raise") -> pd.Series:
        # Deprecated since (pandas) version 1.3.0: Using astype to convert
        # from timezone-naive dtype to timezone-aware dtype is deprecated and
        # will raise in a future version. Use Series.dt.tz_localize() instead.
        series = self._get_pandas_series()
        return series.astype(dtype=dtype, copy=copy, errors=errors)

    def convert_dtypes(
        self,
        infer_objects=True,
        convert_string=True,
        convert_integer=True,
        convert_boolean=True,
        convert_floating=True,
    ) -> pd.Series:
        series = self._get_pandas_series()
        return series.set_flags(
            infer_objects=infer_objects,
            convert_string=convert_string,
            convert_integer=convert_integer,
            convert_boolean=convert_boolean,
            convert_floating=convert_floating,
        )

    def infer_object(self) -> pd.Series:
        series = self._get_pandas_series()
        return series.infer_objects()

    def copy(self, deep=True) -> pd.Series:
        series = self._get_pandas_series()
        return series.copy(deep=deep)

    def bool(self) -> bool:
        series = self._get_pandas_series()
        return series.bool()

    ### Note: to_numpy equivalent already implemented

    def to_period(self, freq=None, copy=True) -> pd.Series:
        series = self._get_pandas_series()
        return series.to_period(freq=freq, copy=copy)

    def to_timestamp(self, freq=None, how="start", copy=True) -> pd.Series:
        series = self._get_pandas_series()
        return series.to_period(freq=freq, how=how, copy=copy)

    def tolist(self) -> list:
        series = self._get_pandas_series()
        return series.tolist()

    to_list = tolist

    def __array__(self, dtype: None) -> np.ndarray:
        series = self._get_pandas_series()
        return np.asarray(series._values, dtype)


class PandasDataFrame(ABC):
    def __init__(self):
        raise Exception("Cannot be instantiated")
