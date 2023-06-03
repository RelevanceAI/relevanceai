def handle_response(response):
    try:
        return response.json()
    except:
        return response.text
