import sys
sys.path.append(".")

import time
from dotenv import load_dotenv
load_dotenv()

from relevanceai import RelevanceAI

client = RelevanceAI()

agent_id = "b9cd9cd5-2aea-45dd-b12a-461a0935528e"

# message_list = [
#     "Research the following company: RelevanceAI relevanceai.com",
#     "Research the following company: Vividly govividly.com",
#     "Research the following company: Airwallex airwallex.com",
#     "Research the following company: Sahha AI sahha.ai",
#     "Research the following company: Relume relume.io",
# ]

# task_ids = [] 

# for message in message_list: 
#     task = client.tasks.trigger_task(agent_id, message)
#     task_ids.append(task.conversation_id)
#     time.sleep(1)

# print(task_ids)

# Given a list of task_ids 
task_ids = ['9e7ed99f-9081-4983-906c-9db800a86478', '88e0a1bf-d1a3-4462-8535-4e0b0da69f4e', '08c87fb3-75cd-4039-b222-def605005aa1', '7b477033-7a6e-4347-ae8e-32ec56ea7cde', '366c496b-74a4-444a-86af-25a1eb023d26']
tool_id = "516c79a3-d097-4336-895d-bf93c2b192fe"

for task_id in task_ids: 
    # approved_task = client.tasks.approve_task(agent_id, task_id)

    # approve only a specific tool (optional) -> use commented code below 
    approved_task = client.tasks.approve_task(agent_id, task_id, tool_id=tool_id)


print("Done")






