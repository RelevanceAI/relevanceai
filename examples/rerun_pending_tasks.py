
"""
Rerun pending tasks. 
"""

import time 
import json
from relevanceai import RelevanceAI
from dotenv import load_dotenv
load_dotenv()

client = RelevanceAI()

agent_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

my_agent = client.agents.retrieve_agent(agent_id=agent_id)

tasks = my_agent.list_tasks(
    max_results=100, 
    state="pending-approval"
)

task_ids = [t.get_id() for t in tasks]

# write task_ids to a json file
with open("task_ids.json", "w") as f:
    json.dump(task_ids, f, indent=4)

# read task_ids to a json file
with open("task_ids.json", "r") as f:
    task_ids = json.load(f)

for t_id in task_ids:
    try: 
        task_rerun = my_agent.rerun_task(conversation_id=t_id,)
        time.sleep(2) 
    except: 
        print("Failed at task_id: " + t_id)
        continue