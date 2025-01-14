import requests

def get_api_definition():
    base_url = "https://api-f1db6c.stack.tryrelevance.com/latest"
    params = {
        "password": "super secret password"
    }
    response = requests.get(f"{base_url}/api_definition", params=params)
    return response.json()["api_definition"]

def find_endpoint(api_definition, handler_name):
    return next((endpoint for endpoint in api_definition if endpoint.get('handler_name') == handler_name), None)

def find_endpoints_containing(api_definition, name_part):
    return [endpoint for endpoint in api_definition 
            if name_part.lower() in endpoint.get('handler_name', '').lower()]

def get_endpoint_schemas(endpoint):
    if not endpoint:
        return None
    
    schemas = {
        'name': endpoint['handler_name'],
        'path': '/'.join(endpoint['http_payload_routing_pattern']['path_parts']),
        'method': endpoint['http_payload_routing_pattern']['method'],
        'input_schema': endpoint.get('validation_schemas', {}).get('input_body_schema', {}),
        'output_schema': endpoint.get('validation_schemas', {}).get('output_body_schema', {})
    }
    return schemas

if __name__ == "__main__":
    try:
        api_definition = get_api_definition()
        
        # Example: Find specific endpoint and get its schemas
        endpoint = find_endpoint(api_definition, "ListStudios")
        if endpoint:
            schemas = get_endpoint_schemas(endpoint)
            print("\nEndpoint Schemas:")
            print(f"Name: {schemas['name']}")
            print(f"Path: {schemas['path']}")
            print(f"Method: {schemas['method']}")
            print("\nInput Schema:")
            print(schemas['input_schema'])
            print("\nOutput Schema:")
            print(schemas['output_schema'])

    except requests.exceptions.RequestException as e:
        print(f"Error fetching API definition: {e}")
