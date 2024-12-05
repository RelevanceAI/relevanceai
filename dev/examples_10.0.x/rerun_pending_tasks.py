
"""
Rerun pending tasks. 
"""

import time 
import json
from relevanceai import RelevanceAI
from dotenv import load_dotenv
load_dotenv()

client = RelevanceAI()

tasks = client.tasks.list_tasks(
    agent_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", 
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
        task_rerun = client.tasks.rerun_task(
            agent_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", 
            conversation_id=t_id,
        )

        time.sleep(2) 
    except: 
        print("Failed at task_id: " + t_id)
        continue
