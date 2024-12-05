
import sys
sys.path.append(".")

from dotenv import load_dotenv
load_dotenv()

from relevanceai import RelevanceAI

client = RelevanceAI()

agent_id = "eaccc0c5-dc01-4553-a4bd-37fb60c99304"
conversation_id = '5f1fdd53-34c2-49b1-a7c3-93ec31e32a38'
task_view = client.tasks.view_task_steps(agent_id, conversation_id)
