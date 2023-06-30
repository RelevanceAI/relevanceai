_SCHEMA_KEYS = {"title", "type", "description"}
_BASE_URL = "https://api-{region}.stack.tryrelevance.com/latest/studios/"
_RANDOM_ID_WARNING = """
Your studio id is randomly generated. 
To ensure you are updating the same chain you should specify the id on `rai.create(id=id)` """
_LOW_CODE_NOTEBOOK_BLURB = """
=============Low Code Notebook================
You can share/visualize your chain as an app in our low code notebook here: 
https://chain.relevanceai.com/notebook/{region}/{project}/{id}/app

===============With Requests==================
Here is an example of how to run the chain with API: 
import requests
requests.post(
    https://api-{region}.stack.tryrelevance.com/latest/studios/{id}/trigger_limited", 
    json={{
        "project": "{project}",
        "params": {{
            YOUR PARAMS HERE
        }}
    }}
)

===============With Python SDK================      
Here is an example of how to run the chain with Python: 

```python
import relevanceai as rai

chain = rai.load("{id}")
chain.run({{YOUR PARAMS HERE}})
```
            
"""
