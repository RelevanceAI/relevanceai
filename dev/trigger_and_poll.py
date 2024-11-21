
import sys
sys.path.append(".")
from textwrap import dedent
import time 
from dotenv import load_dotenv
load_dotenv()

from relevanceai import RelevanceAI

client = RelevanceAI()

agent_id = "b9cd9cd5-2aea-45dd-b12a-461a0935528e"

message = dedent(f"""
Research the following company:        
RelevanceAI relevanceai.com      
""")

task = client.tasks.trigger_task(
    agent_id=agent_id,
    message=message
)

conversation_id = task.conversation_id

while not client.tasks.get_task_output_preview(agent_id, conversation_id): 
    print("polling...\n")
    time.sleep(5)

task_output_preview = client.tasks.get_task_output_preview(agent_id, conversation_id)

print(task_output_preview["answer"])
