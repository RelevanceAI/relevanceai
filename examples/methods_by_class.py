
"""
Methods by class. 
"""

### Getting started

from relevanceai import RelevanceAI
from dotenv import load_dotenv
load_dotenv()

client = RelevanceAI()

### Agents 

client.agents.list_agents()

client.agents.retrieve_agent(agent_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")

client.agents.delete_agent(agent_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")

### Tasks 

client.tasks.list_tasks(agent_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")

client.tasks.retrieve_task(
    agent_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    conversation_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
)

client.tasks.view_task_steps(
    agent_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    conversation_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
)

client.tasks.approve_task(
    agent_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    conversation_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    tool_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
)

client.tasks.delete_task(
    agent_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    conversation_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
)

message = "OUR_SALES_EMAIL```xxxxxx@xxxxxx.com```EMAIL_THREAD_ID```xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx```\nLinkedIn_Chat_ID```xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx```"

client.tasks.trigger_task(
    agent_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", 
    message=message
)

client.tasks.rerun_task(
    agent_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", 
    conversation_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
)

client.tasks.schedule_action_in_task(
    agent_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", 
    conversation_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", 
    message=message,
    minutes_until_schedule=86400 # 1 day 
)

### Tools

client.tools.list_tools() 

client.tools.retrieve_tool(tool_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")

client.tools._get_params_as_json_string(tool_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")

client.tools._get_steps_as_json_string(tool_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")

params = {
    "text": "Yes",
    "long_text": "This is a long text",
}

client.tools.trigger_tool(
    tool_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    params=params
)

### Knowledge 

client.knowledge.list_knowledge()

client.knowledge.retrieve_knowledge(knowledge_set="xxxxxx") # put the table name here

client.knowledge.delete_knowledge(knowledge_set="xxxxxx")