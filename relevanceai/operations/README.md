# Operations

Going forward, operations to a datatset will be supported in the following ways

```
operator = {operation}ops(*args, **kwargs)

operation.forward(dataset: Dataset, *args, **kwargs)
```

## OR

```
dataset.{operation}(alias: str, *args, **kwargs)
```

## Imports

All of following examples a written with the following imports in mind.

```
from relevanceai import Client

from relevanceai import Dataset

client = Client()

dataset = client.Dataset(dataset_id)
```

## Example applying Clustering to a Dataset

### `ClusterOps`

```
operator = client.ClusterOps(
    model: str,
    n_clusters: int,
    alias: str,
    config: Dict[str, Any], # config is passed through underlying model implementation
)

operator.forward(
    dataset: Dataset, # users must initialise a Dataset first
    vector_fields: List[str],
)
```

### `dataset.cluster`

```
dataset.cluster(
    model: str,
    n_clusters: int,
    alias: str,
    vector_fields:
    List[str],
    config: Dict[str, Any]
)
```

## Example applying Dimensionality Reduction to a Dataset

### `DROps`

```
operator = client.DROps(
    model: str,
    n_dimensions: int,
    alias: str,
    config: Dict[str, Any], # config is passed through underlying model implementation
)

operator.forward(
    dataset: Dataset, # users must initialise a Dataset first
    vector_fields: List[str],
)
```

### `dataset.dr`

```
dataset.dr(
    model: str,
    n_clusters: int,
    alias: str,
    vector_fields:
    List[str],
    config: Dict[str, Any]
)
```

## Example applying Vectorize to a Dataset

### `VectorOps`

```
operator = client.VectorOps(
    model: str,
    alias: str,
    config: Dict[str, Any], # config is passed through underlying model implementation
)
```
to perform
```
operator(
    dataset: Dataset, # users must initialise a Dataset first
    vector_fields: List[str],
)
```
or
```
operator.forward(
    dataset: Dataset, # users must initialise a Dataset first
    vector_fields: List[str],
)
```

### `dataset.vectorize`

```
dataset.vectorize(
    model: str,
    n_clusters: int,
    alias: str,
    vector_fields:
    List[str],
    config: Dict[str, Any]
)
```
