import numpy as np


def numpy_document(_id: str):
    return {
        "_id": _id,
        "sample_1_numpy": np.random.randint(5, size=1)[0],
        "sample_2_numpy": np.random.rand(3, 2),
        "sample_3_numpy": np.nan,
    }
