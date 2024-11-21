"""
Trigger and poll for task outputs of agents

"""

from textwrap import dedent
import time 
from dotenv import load_dotenv
load_dotenv()

from relevanceai import RelevanceAI

client = RelevanceAI()

agent_id = "xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

message = dedent(f"""
Research the following company:        
RelevanceAI relevanceai.com      
""")

task = client.tasks.trigger_task(
    agent_id=agent_id,
    message=message
)

while not client.tasks.get_task_output_preview(agent_id, task.conversation_id): 
    print("polling...\n")
    time.sleep(5)

task_output_preview = client.tasks.get_task_output_preview(agent_id, task.conversation_id)

print(task_output_preview["answer"])
