# Different base clasess:
1. `transform_base.py`: This is where all the checks and non dataset interactive functions go. <- `transform.py` should inherit this
2. `ops_run.py`: This is where all the dataset related interactions goes. 
3. `ops_api_base.py`: This is when API client methods are required. <- `ops.py` should inherit this

# How to create a Operation & Transform
You need two key things
`transform.py`: This will take documents as inputs run the necessary transformation and return updated part of the documents as outputs.
`ops.py`: This will take a transform and run it against a whole dataset.