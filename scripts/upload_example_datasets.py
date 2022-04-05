from typing import List

from relevanceai import Client
from relevanceai.utils.datasets import ExampleDatasets


def determine_datasets(client: Client) -> frozenset:
    """
    Determine which example datasets to upload. If the dataset already
    exists in the account, that dataset will be ignored.
    """
    current_datasets = set(client.datasets.list()["datasets"])
    example_datasets = set(ExampleDatasets().list_datasets())
    return frozenset((current_datasets ^ example_datasets) - current_datasets)


def download_example_dataset(name: str) -> List[dict]:
    print(f"Downloading {name}...")
    dataset = ExampleDatasets().get_dataset(
        name, number_of_documents=1_000_000  # Ensure all documents are downloaded
    )
    print(f"Finished downloading {name}.")

    return dataset


def upload_datasets() -> None:
    client = Client(region="us-east-1")

    example_datasets = determine_datasets(client)

    if example_datasets:
        print(
            "The following datasets will be uploaded:"
            + "".join(map(lambda dataset: f"\n  * {dataset}", example_datasets))
            + "\n"
        )
        for name in example_datasets:
            client.create_dataset(name)
            print(f"Created {name} in {client.project}.")
            ds = client.Dataset(name)
            try:
                ds.insert_documents(download_example_dataset(name), create_id=True)
            except Exception as e:
                # If upload is unsuccessful, delete datataset from project
                print(
                    f"Unable to upload {name}. "
                    + f"Removing {name} from {client.project}."
                )
                client.datasets.delete(name)
                raise e
    else:
        print(f"All example datasets already exists in {client.project}.")


if __name__ == "__main__":
    upload_datasets()
