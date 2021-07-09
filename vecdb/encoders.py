from .base import Base

class Encoders(Base):
    def __init__(self, project: str, api_key: str, base_url: str):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url

    def textimage(self, text: str):
        return self.make_http_request(
            "services/encoders/textimage", 
            method="POST",
            parameters={
                "text": text
            })
    
    def text(self, text: str):
        return self.make_http_request(
            "services/encoders/text",
            method="POST",
            parameters={
                "text": text
            })
