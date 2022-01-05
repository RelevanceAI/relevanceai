from relevanceai import Client

PROJECT = 'josephtwins'
API_KEY = 'dmllM2ZId0J1M3VVTEcyQTBmVXY6QTlCZE9GdjlTNFN0WlZ3ckNmZ0Fudw'
DATASET_ID = 'set6_team_comps_200'


def main():

    client = Client(
        project=PROJECT,
        api_key=API_KEY,
    )

    clusterer = client.projector.dendrogram(
        dataset_id=DATASET_ID,
        vector_fields=['comp_vector_'],
        node_label='comp_name',
        width=1800,
        height=850
    )


if __name__ == '__main__':
    main()
