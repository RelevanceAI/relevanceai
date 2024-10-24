## Relevance AI - The home of your AI Workforce

🔥 Use Relevance to build AI agents for your AI workforce

[Sign up for a free account ->](https://app.relevanceai.com)

## 🧠 Documentation

| Type      | Link |
| ------------- | ----------- |
| Home Page | [Home Page](https://relevanceai.com/) |
| Platform | [Platform](https://app.relevanceai.com/) |
| Developer Documentation | [Documentation](https://sdk.relevanceai.com/) |

# Relevance AI SDK

Welcome to the Relevance AI SDK! This guide will help you set up and start using the SDK to interact with your AI agents, tools, and knowledge.

## Installation

To get started, you'll need to install the RelevanceAI library in a Python 3 environment. Run the following command in your terminal:

```bash
pip install relevanceai
```

## Create an Account

Before using the SDK, ensure you have an account with Relevance AI.

1. Sign up for a free account at [Relevance AI](https://app.relevanceai.com) and log in.
2. Create a new secret key at [SDK Login](https://app.relevanceai.com/login/sdk). Scroll to the bottom of the integrations page, click on "+ Create new secret key," and select "Admin" permissions.

## Set Up Your Client

To interact with Relevance AI, you'll need to set up a client. Start by importing the library:

```python
from relevanceai import RelevanceAI
client = RelevanceAI()
```

### Validate Client Credentials

You can validate your client credentials by storing them as environment variables and loading them into your project using `python-dotenv` or the `os` library:

```env
RAI_API_KEY=
RAI_REGION=
RAI_PROJECT=
```

```python
from dotenv import load_dotenv
load_dotenv()

from relevanceai import RelevanceAI
client = RelevanceAI()
```

Alternatively, pass the credentials directly to the client:

```python
from relevanceai import RelevanceAI
client = RelevanceAI(
    api_key="your_api_key", 
    region="your_region", 
    project="your_project"
)
```

You are now ready to start using Relevance AI via the Python SDK.

## Quickstart

### Using Agents & Tasks

List all the agents in your project:

```python
from relevanceai import RelevanceAI
client = RelevanceAI()

my_agents = client.agents.list_agents()
print(my_agents)
```

Retrieve and interact with a specific agent:

```python
my_agent = client.agents.retrieve_agent(agent_id="xxxxxxxx")

message = "Let's qualify this lead:\n\nName: Ethan Trang\n\nCompany: Relevance AI\n\nEmail: ethan@relevanceai.com"

triggered_task = client.tasks.trigger_task(
    agent_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", 
    message=message
)
print(triggered_task)
```

### Using Tools

List all the tools in your project:

```python
my_tools = client.tools.list_tools()
print(my_tools)
```

Retrieve and interact with a specific tool:

```python
my_tool = client.tools.retrieve_tool(tool_id="xxxxxxxx")

params = {"text": "This is text", "number": 245}

tool_response = client.tools.trigger_tool(
    tool_id="xxxxxxxx",
    params=params
)
print(tool_response)
```

## Explore More

Explore all the methods available for agents, tasks, tools, and knowledge with the [documentation](https://sdk.relevanceai.com/)
