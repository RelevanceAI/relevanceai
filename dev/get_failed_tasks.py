import sys
sys.path.append(".")

###* Getting started

from relevanceai import RelevanceAI
from dotenv import load_dotenv
load_dotenv()

client = RelevanceAI()

tasks = client.tasks.list_tasks()

print(tasks)
