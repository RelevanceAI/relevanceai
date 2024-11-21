import requests
import json 
import subprocess  # Import subprocess module

response = requests.get('https://api-1e3042.stack.tryrelevance.com/latest/api_definition?password=super%20secret%20password')
api_definitions = response.json()["api_definition"]

for api in api_definitions: 

    if api["handler_name"] == "ListConversationStudiosHistory": 

        # input_body_schema = api["validation_schemas"]["input_body_schema"]
        output_body_schema = api["validation_schemas"]["output_body_schema"]
        with open('mydev/output_body_schema.json', 'w') as json_file:
            json.dump(output_body_schema, json_file, indent=4)

        subprocess.run(
            [
                'datamodel-codegen', 
                '--input', 'mydev/output_body_schema.json', 
                '--output', 'mydev/output_body_model.py', 
                '--output-model-type', 'pydantic_v2.BaseModel'
            ]
        )

        print(api["handler_name"])
    continue

"""
python mydev/get_api_definition.py
"""
