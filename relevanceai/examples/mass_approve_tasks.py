
"""
Mass approve tasks. 
"""

import time

from relevanceai import RelevanceAI
from dotenv import load_dotenv
load_dotenv()

client = RelevanceAI()

agent_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

tasks_list = client.tasks.list_tasks(
    agent_id=agent_id, 
    max_results=1000, 
    state="pending-approval"
)

conversation_id_list = [task.get_id() for task in tasks_list]

for conversation_id in conversation_id_list: 
    client.tasks.approve_task(
        agent_id=agent_id, 
        conversation_id=conversation_id
    )
    time.sleep(3)
