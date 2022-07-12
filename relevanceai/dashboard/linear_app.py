import requests
import datetime
def preprocess_linear_doc(doc):
    doc['_id'] = doc['id']
    created_at = datetime.datetime.strptime(doc['createdAt'], "%Y-%m-%dT%H:%M:%S.%fZ")
    
    doc['started'] = 0    
    if doc['startedAt']:
        started_at = datetime.datetime.strptime(doc['startedAt'], "%Y-%m-%dT%H:%M:%S.%fZ")
        doc['started'] = 0
        doc['timeBetweenCreateToStart'] = (started_at - created_at).seconds//3600
        
    doc['completed'] = 0
    if doc['completedAt']:
        completed_at = datetime.datetime.strptime(doc['completedAt'], "%Y-%m-%dT%H:%M:%S.%fZ")
        doc['completed'] = 1
        if doc['startedAt']:
            doc['timeBetweenStartToComplete'] = (completed_at - started_at).seconds//3600
        doc['timeBetweenCreateToComplete'] = (completed_at - created_at).seconds//3600
    
    doc['canceled'] = 0
    if doc['canceledAt']:
        completed_at = datetime.datetime.strptime(doc['canceledAt'], "%Y-%m-%dT%H:%M:%S.%fZ")
        doc['canceled'] = 1
    
    if not doc['canceled'] and not doc['completed']:
        doc['timePastCreate'] = (datetime.datetime.now() - created_at).seconds//3600
        if doc['startedAt']:
            doc['timePastStart'] = (datetime.datetime.now() - started_at).seconds//3600
    if doc['dueDate']:
        due_date = datetime.datetime.strptime(doc['dueDate'], "%Y-%m-%d")
        if doc['completedAt']:
            doc['timeBetweenDueToComplete'] = (completed_at - due_date).seconds//3600
        elif not doc['canceled']:
            doc['timePastDue'] = (datetime.datetime.now() - due_date).seconds//3600
    doc['numLabels'] = len(doc['labels']['nodes'])
    doc['noLabels'] = 0
    if doc['numLabels'] == 0:
        doc['noLabels'] = 1
    return doc

def linear_recipe(dataset_id, api_key=api_key):
    docs = []
    next_page = True

    query_string = """
        query Issues($orderBy: PaginationOrderBy) {{
          issues(first: 100, orderBy: $orderBy {after}) {{
            edges {{
              node {{
                id
                createdAt
                updatedAt
                archivedAt
                number
                title
                description
                project {{
                  name
                }}
                priority
                priorityLabel
                estimate
                startedAt
                completedAt
                canceledAt
                dueDate
                team {{
                  name
                }}
                creator {{
                  name
                }}
                assignee {{
                  name
                }}
                url
                labels {{
                  nodes {{
                    name
                  }}
                }}
              }}
            }}
            pageInfo {{
              hasNextPage
              endCursor
            }}
          }}
        }}
    """
    print("Downloading data from Linear")
    results = requests.post(
            "https://api.linear.app/graphql",
            headers = {
                "Authorization" : f"Bearer {api_key}"
            },
            json = {
                'query' : query_string.format(after="")
            }
        ).json()
    docs += [preprocess_linear_doc(n['node']) for n in results['data']['issues']['edges']]
    next_page = results['data']['issues']['pageInfo']['hasNextPage']

    while next_page:
        results = requests.post(
            "https://api.linear.app/graphql",
            headers = {
                "Authorization" : f"Bearer {api_key}"
            },
            json = {
                'query' : query_string.format(after=f', after: "{results["data"]["issues"]["pageInfo"]["endCursor"]}"')
            }
        ).json()
        docs += [preprocess_linear_doc(n['node']) for n in results['data']['issues']['edges']]
        next_page = results['data']['issues']['pageInfo']['hasNextPage']
    print(f"Inserting {len(docs)} linear issues to Relevance")
    ds = client.Dataset(dataset_id)
    ds.insert_documents(docs, create_id=False)
    print("Generating App for exploration & insight")
    ds.create_metrics_chart_app(
        app_name="Linear Dashboard", 
        metrics=[
            "priority", 
            "timeBetweenCreateStart", 
            "timeBetweenStartToComplete", 
            "timePastDue",
            "timePastStart",
            "timePastCreate",
            "canceled",
            "completed"
        ],
        groupby=[
            "assignee.name", "creator.name", 
            "team.name", "labels.nodes.name",
            "priorityLabel"
        ],
        sort=["completed"]
    )
    return docs