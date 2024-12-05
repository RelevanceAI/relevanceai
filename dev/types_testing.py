import sys 
sys.path.append(".") 

from dotenv import load_dotenv
load_dotenv()

from relevanceai import RelevanceAI 

client = RelevanceAI() 

agent_id = 'b8d38c82-fca4-45ac-8699-bfa4103d9faa'

tasks = client.tasks.list_tasks(agent_id=agent_id)

print(f"Number of tasks: {len(tasks)}")

for t in tasks: 
    print(t.get_id())
