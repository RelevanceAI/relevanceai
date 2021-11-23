from relevanceai.base import Base


class Aggregate(Base):
    """Aggregate service"""

    def __init__(self, project, api_key, base_url):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(project, api_key, base_url)

    def aggregate(
        self,
        dataset_id: str,
        metrics: list = [],
        groupby: list = [],
        filters: list = [],
        page_size: int = 20,
        page: int = 1,
        asc: bool = False,
        flatten: bool = True,
        alias: str = "default",
    ):
        """
        Aggregation/Groupby of a collection using an aggregation query. The aggregation query is a json body that follows the schema of:
        
        >>> {
        >>>        "groupby" : [
        >>>            {"name": <alias>, "field": <field in the collection>, "agg": "category"},
        >>>            {"name": <alias>, "field": <another groupby field in the collection>, "agg": "numeric"}
        >>>        ],
        >>>        "metrics" : [
        >>>            {"name": <alias>, "field": <numeric field in the collection>, "agg": "avg"}
        >>>            {"name": <alias>, "field": <another numeric field in the collection>, "agg": "max"}
        >>>        ]
        >>>    }
        >>>    For example, one can use the following aggregations to group score based on region and player name.
        >>>    {
        >>>        "groupby" : [
        >>>            {"name": "region", "field": "player_region", "agg": "category"},
        >>>            {"name": "player_name", "field": "name", "agg": "category"}
        >>>        ],
        >>>        "metrics" : [
        >>>            {"name": "average_score", "field": "final_score", "agg": "avg"},
        >>>            {"name": "max_score", "field": "final_score", "agg": "max"},
        >>>            {'name':'total_score','field':"final_score", 'agg':'sum'},
        >>>            {'name':'average_deaths','field':"final_deaths", 'agg':'avg'},
        >>>            {'name':'highest_deaths','field':"final_deaths", 'agg':'max'},
        >>>        ]
        >>>    }

        "groupby" is the fields you want to split the data into. These are the available groupby types:

            - category : groupby a field that is a category
            - numeric: groupby a field that is a numeric

        "metrics" is the fields and metrics you want to calculate in each of those, every aggregation includes a frequency metric. These are the available metric types:
            
            - "avg", "max", "min", "sum", "cardinality"

        The response returned has the following in descending order. \n

        If you want to return documents, specify a "group_size" parameter and a "select_fields" parameter if you want to limit the specific fields chosen. This looks as such:
        
        >>>    {
        >>>    'groupby':[
        >>>        {'name':'Manufacturer','field':'manufacturer','agg':'category',
        >>>        'group_size': 10, 'select_fields': ["name"]},
        >>>    ],
        >>>    'metrics':[
        >>>        {'name':'Price Average','field':'price','agg':'avg'},
        >>>    ],
        >>>    }
        >>>
        >>>    {"title": {"title": "books", "frequency": 200, "documents": [{...}, {...}]}, {"title": "books", "frequency": 100, "documents": [{...}, {...}]}}
        
        For array-aggregations, you can add "agg": "array" into the aggregation query.
        
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        metrics: list
            Fields and metrics you want to calculate
        groupby: list 
            Fields you want to split the data into
        filters: list
            Query for filtering the search results
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        asc: bool
            Whether to sort results by ascending or descending order
        flatten: bool
            Whether to flatten
        alias: string
            Alias used to name a vector field. Belongs in field_{alias} vector
        """
        return self.make_http_request(
            "/services/aggregate/aggregate",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "aggregation_query": {"groupby": groupby, "metrics": metrics},
                "filters": filters,
                "page_size": page_size,
                "page": page,
                "asc": asc,
                "flatten": flatten,
                "alias": alias,
            },
        )
