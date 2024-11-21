
#? required to test from mydev
import sys
sys.path.append(".")

###* Getting started

from relevanceai import RelevanceAI
from dotenv import load_dotenv
load_dotenv()

client = RelevanceAI()

###* Agents 

client.agents.list_agents()

client.agents.retrieve_agent(agent_id="91e5b24f-a89b-4c53-87cd-19cfb3cfd946")

client.agents.delete_agent(agent_id="...")

# todo: list tools, subagents
#? some methods we don't want to users to have access to

###* Tasks 

client.tasks.list_tasks(agent_id="91e5b24f-a89b-4c53-87cd-19cfb3cfd946")

client.tasks.view_task_steps(
    agent_id="91e5b24f-a89b-4c53-87cd-19cfb3cfd946",
    conversation_id="6616abff-2e18-4390-b66d-4895bb6bfae5"
)

client.tasks.approve_task(
    agent_id="98db5752-cd35-45d3-875a-c31c81db9328",
    conversation_id="91aae387-1318-45be-a991-1acf0cfd7ba4",
    tool_id="a788e045-8057-4265-95e1-0004eb0101b9"
)

client.tasks.retrieve_task(
    agent_id="91e5b24f-a89b-4c53-87cd-19cfb3cfd946",
    conversation_id="6616abff-2e18-4390-b66d-4895bb6bfae5"
)

client.tasks.delete_task(
    agent_id="91e5b24f-a89b-4c53-87cd-19cfb3cfd946",
    conversation_id="..."
)

message = "OUR_SALES_EMAIL```ethan.trang5521@gmail.com```EMAIL_THREAD_ID```191c00ca5311b7db```\nLinkedIn_Chat_ID```6snYpkCdUrOu5Or3JpMTNw```"

client.tasks.trigger_task(
    agent_id="91e5b24f-a89b-4c53-87cd-19cfb3cfd946", 
    message=message
)

# todo: way to get response

client.tasks.rerun_task(
    agent_id="91e5b24f-a89b-4c53-87cd-19cfb3cfd946", 
    conversation_id="0c07dec5-a952-4304-9a1a-83467757771c"
)

client.tasks.schedule_action_in_task(
    agent_id="91e5b24f-a89b-4c53-87cd-19cfb3cfd946", 
    conversation_id="6616abff-2e18-4390-b66d-4895bb6bfae5", 
    message=message,
    minutes_until_schedule=86400 # 1 day 
)

###* Tools

client.tools.list_tools() 

client.tools.retrieve_tool(tool_id="cdf61ebd-53a1-4f78-9b6f-96a720a0114a")

client.tools._get_params_as_json_string(tool_id="cdf61ebd-53a1-4f78-9b6f-96a720a0114a")

client.tools._get_steps_as_json_string(tool_id="cdf61ebd-53a1-4f78-9b6f-96a720a0114a")

params = {
    "text": "Yes",
    "long_text": "This is a long text",
}

client.tools.trigger_tool(
    tool_id="cdf61ebd-53a1-4f78-9b6f-96a720a0114a",
    params=params
)


# todo: transform steps 

###* Knowledge 

client.knowledge.list_knowledge()

client.knowledge.retrieve_knowledge(knowledge_set="")

client.knowledge.delete_knowledge(knowledge_set="")

###* Integrations 

###* Analytics 

###* Other cool stuff

# rerunning 89 tasks 

# get api definitions to pydantic models 