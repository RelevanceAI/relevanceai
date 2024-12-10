
"""
Mass approve tasks. 
"""

import time

from relevanceai import RelevanceAI
from dotenv import load_dotenv
load_dotenv()

client = RelevanceAI()

agent_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

my_agent = client.agents.retrieve_agent(agent_id=agent_id)

tasks_list = my_agent.list_tasks(
    max_results=1000, 
    state="pending-approval"
)

conversation_id_list = [task.get_id() for task in tasks_list]

for conversation_id in conversation_id_list: 
    my_agent.approve_task(conversation_id=conversation_id)
    time.sleep(3)