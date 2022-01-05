from relevanceai import Client

PROJECT = 'josephtwins'
API_KEY = 'dmllM2ZId0J1M3VVTEcyQTBmVXY6QTlCZE9GdjlTNFN0WlZ3ckNmZ0Fudw'
DATASET_ID = 'set6_team_comps_50'


def main():

    client = Client(
        project=PROJECT,
        api_key=API_KEY,
    )

    cluster_list = client.services.cluster.list(
        dataset_id=DATASET_ID,
        vector_field='comp_vector'
    )

    print(cluster_list)

if __name__ == '__main__':
    main()