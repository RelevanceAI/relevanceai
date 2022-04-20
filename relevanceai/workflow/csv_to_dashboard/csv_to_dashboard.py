from relevanceai import Client
from relevanceai.workflow.sequential import SequentialWorkflow

from relevanceai.workflow.sequential import Input, Output

from relevanceai.operations import ClusterOps, VectorizeOps

client = Client(
    token="59066979f4876d91beea:aFdYdnZYNEJ5XzFjUWdIem9KbTk6cFNmRlhpdVdSdmlsekVrTE83VC1CUQ:us-east-1:wBfUxxaHI3QsutqMzqliyGr9RbC3"
)
ds = client.Dataset("_mock_dataset_")

workflow = SequentialWorkflow(
    [
        Input([field for field in ds.schema], chunksize=50),
        VectorizeOps(client.credentials),
        ClusterOps(client.credentials),
        Output("simple"),
    ],
    log_filename="logs",
)

workflow.run(ds, verbose=True)
