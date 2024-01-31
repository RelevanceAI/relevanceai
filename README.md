## Relevance AI - The home of your AI Workforce

🔥 Use Relevance to build AI agents for your AI workforce:
- ⚡ Connect your python api's to tools for Agents or custom actions for GPTs.
- 🚀 Share your tools as AI web apps with your team to use.

[Sign up for a free account ->](https://app.relevanceai.com)

## 🧠 Documentation

| Type      | Link |
| ------------- | ----------- |
| Home Page | [Home Page](https://relevanceai.com/) |
| Platform | [Platform](https://app.relevanceai.com/) |
| Developer Documentation | [Documentation](https://sdk.relevanceai.com/) |

## Getting Started

1. Installation:
`pip install relevanceai`

This example uses fastapi and uvicorn so lets install that too:
`pip install fastapi`
`pip install uvicorn[standard]`

2. Create your FastAPI app - *skip if you already have one*
Here is a quick example of a FastAPI app:
```python
from fastapi import FastAPI
app = FastAPI()

class HelloWorldParams(BaseModel):
    message : str = Query(..., title="Message", description="message from user")

class HelloWorldResponse(BaseModel):
    reply : str

def hello_world(prompt):
    return {"reply" : "hello world"}

@app.post("/hello_world", name="Hello World", description="Reply always with hello world", response_model=HelloWorldResponse)
def hello_world_api(commons: HelloWorldParams):
    return hello_world(commons.message)
```

3. Describe for your tools
Make sure to give your FastAPI endpoints as much descrition as possible. These provided descriptions are utilized in the agent prompt so that the Agent can better understand your tools.

For example:
Add a `title` and `description` for the inputs of your tool, explaining what they are and what kind of value to provide:
```python
class HelloWorldParams(BaseModel):
    message : str = Query(..., description="message from user")
```
Add a `name` and `description` about the tool explaining when to use it and what it does:
```python
@app.post("/hello_world", name="Hello World", description="Reply always with hello world", response_model=HelloWorldResponse)
```
Relevance AI will automatically take these values from your fastapi app and use it to create a prompt for the agent.


4. Lets connect it live to Relevance AI
In short all it takes to connect is to add the following lines to your app:
```python
from relevanceai.connect.fastapi import connect_fastapi_to_rai

connect_fastapi_to_rai(app.routes, PUBLIC_URL)
```
Where `PUBLIC_URL` is the public url of your app. For example `https://myapp.com`.

If you are working locally and dont have a public url you can use [ngrok](https://ngrok.com/) to create a public url for your app.

5. All together
```python
from pyngrok import ngrok
PUBLIC_URL = ngrok.connect(8000).public_url
```

5. Putting this all together
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List

#create FastAPI app
app = FastAPI()
#add cors middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class HelloWorldParams(BaseModel):
    message : str = Query(..., description="message from user")

class HelloWorldResponse(BaseModel):
    reply : str

def hello_world(prompt):
    return {"reply" : "hello world"}

@app.post(
        "/hello_world", name="Hello World", description="Reply always with hello world", response_model=HelloWorldResponse
    )
def hello_world_api(commons: HelloWorldParams):
    return hello_world(commons.message)

#If you are deploying the api from a local computer use ngrok to expose a public url.
from pyngrok import ngrok
PUBLIC_URL = ngrok.connect(8000).public_url

#This will create a Tool in Relevance AI that will call your API endpoint
from relevanceai.connect.fastapi import connect_fastapi_to_rai
connect_fastapi_to_rai(app.routes, PUBLIC_URL)
```

## Roadmap & Contribution
More examples and api connectors coming soon. Feel free to contribute to this repo.