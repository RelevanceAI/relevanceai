# -*- coding: utf-8 -*-
from vecdb.base import Base


class Encoders(Base):
    def __init__(self, project: str, api_key: str, base_url: str):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url

    def textimage(self, text: str):
        return self.make_http_request(
            endpoint="services/encoders/textimage",
            method="GET",
            parameters={"text": text},
        )

    def text(self, text: str):
        return self.make_http_request(
            endpoint="services/encoders/text", method="GET", parameters={"text": text}
        )

    def multi_text(self, text):
        """Encode Multilingual text"""
        return self.make_http_request(
            endpoint="services/encoders/multi_text",
            method="GET",
            parameters={"text": text},
        )
