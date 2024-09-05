## Relevance AI - The home of your AI Workforce

🔥 Use Relevance to build AI agents for your AI workforce

[Sign up for a free account ->](https://app.relevanceai.com)

## 🧠 Documentation

| Type      | Link |
| ------------- | ----------- |
| Home Page | [Home Page](https://relevanceai.com/) |
| Platform | [Platform](https://app.relevanceai.com/) |
| Developer Documentation | [Documentation](https://sdk.relevanceai.com/) |

## Installation

Install the Relevance AI SDK using pip:

```bash
pip install relevanceai
```

## Getting Started

### 1. Instantiate the Client

To get started, you'll need to instantiate the Relevance AI client using your API key, region, and project ID. Ensure you load your environment variables securely.

```python
from relevanceai import RelevanceAI

client = RelevanceAI()
```

### 2. Working with Agents

The SDK allows you to manage agents, retrieve agent details, and interact with agent tools.

#### List Agents

```python
client.beta.agents.list()
```

#### Retrieve Agent Details

```python
client.beta.agents.retrieve("agent_id")
```

#### List Agent Tools

```python
client.beta.agents.list_tools("agent_id")
```

### 3. Task Management

Tasks are central to automating workflows with your AI agents. You can trigger tasks, manage conversations, and more.

#### Working with Tasks

```python
client.beta.tasks.trigger(
    agent_id="your_agent_id",
    message={
        "role": "user",
        "content": "Write me a blog post"
    },
)
```

#### List Conversations

```python
client.beta.tasks.list_conversation(
    agent_id="your_agent_id",
    conversation_id="your_conversation_id",
)
```

## Documentation

For more detailed documentation, please visit:

- [Home Page](https://relevanceai.com/)
- [Platform](https://app.relevanceai.com/)
- [Developer Documentation](https://sdk.relevanceai.com/)

## Contributing

We welcome contributions! Feel free to submit pull requests and report issues. More examples and connectors will be added soon.

## License

This project is licensed under the terms of the MIT license.