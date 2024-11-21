##* TESTING WITH REQUEST LIB

import requests



# def handle_response(response):
#     try:
#         return response.json()
#     except:
#         return response.text
    
# def load(id):
#     """loads a chain via id"""

#     response = requests.get(
#         f"https://api-{region}.stack.tryrelevance.com/latest/studios/{id}/get",
#         headers=headers,
#     )
#     res = handle_response(response)
#     # tool = Tool(name="", description="", parameters={}, id=id, auth=auth)
#     return res


# def trigger_tool(studio_id): 
#     json = {
#         "params": {
#             "options": "Yes"
#         }
#     }

#     response = requests.post(
#         f"https://api-f1db6c.stack.tryrelevance.com/latest/studios/{studio_id}/trigger", 
#         headers=headers, 
#         json=json,
#     )

#     res = handle_response(response)

#     return res

# # studio_id = load(id=id)["studio"]["studio_id"]
# res = trigger_tool("d6e98f91-8caa-4a80-b608-149e07de6038")

### Mass approval of tasks 

# Get list of pending task ids

import requests
import sys
sys.path.append(".")
from relevanceai import RelevanceAI 

region = "d7b62b"
project = "11c518f130b2-4273-a714-887a25399a9f"
api_key="sk-MzgxMGVmMTgtMzg4OC00YmI1LWJmZWItNjI3OWVlOGE5ZjJh"
headers = {"Authorization": f"{project}:{api_key}"}

agent_id="9fecf8a2-bfa4-4461-83bc-ef0f04cba70f"

# client = RelevanceAI(
#     api_key=api_key,
#     project=project,
#     region=region
# )

# knowledge_set_list = client.tasks.list_tasks(
#     agent_id=agent_id, 
#     max_results=4000, 
#     state="pending-approval"
# )

# knowledge_set_list

import json 

import time 
with open("mydev/papershift_tasks.json", "r") as f: 
    knowledge_set_list = json.load(f)

tasks_ran = 0
tasks_id_ran = 0

for knowledge_set in knowledge_set_list: 
    time.sleep(3)
    print(tasks_ran)

    response = requests.post(
        f"https://api-{region}.stack.tryrelevance.com/latest/agents/{agent_id}/tasks/{knowledge_set}/view",
        headers=headers,
        json={
            "page_size": 100,
        }
    )

    # For each task see if it requires confirmation and if it is the send email tool, the approve the task 

    result: list = response.json()["results"]

    for r in result: 

        try:
            if r['content']['requires_confirmation'] == True and r['content']['tool_config']['id'] == '7d4d5c2e-0e19-49ce-83d6-a42dc444ff76': # send email tool id 
                print(f"Approving task {knowledge_set}")

                action = r["content"]["action_details"]["action"]
                action_request_id = r["content"]["action_details"]["action_request_id"]

                trigger_response = requests.post(
                    f"https://api-{region}.stack.tryrelevance.com/latest/agents/trigger",
                    headers=headers,
                    json={
                        "agent_id": f"{agent_id}",
                        "conversation_id": f"{knowledge_set}",
                        "message": {
                            "action": f"{action}",
                            "action_request_id": f"{action_request_id}",
                            "action_params_override": {},
                            "role": "action-confirm"
                        },
                        "debug": True,
                        "is_debug_mode_task": False
                    }
                )

                tasks_ran += 1
        except Exception as e: 
            print(f"Error approving task {knowledge_set}: {e}")

print(f"Tasks ran: {tasks_ran}")
print(f"Task ids ran : {len(knowledge_set_list) - tasks_ran}")