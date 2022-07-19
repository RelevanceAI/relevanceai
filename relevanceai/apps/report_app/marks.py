import warnings
from relevanceai.apps.report_app.base import ReportBase


class ReportMarks(ReportBase):
    def bold(self, content):
        return [{"type": "text", "text": content, "marks": [{"type": "bold"}]}]

    def italic(self, content):
        return [{"type": "text", "text": content, "marks": [{"type": "italic"}]}]

    def strike(self, content):
        return [{"type": "text", "text": content, "marks": [{"type": "strike"}]}]

    def underline(self, content):
        return [{"type": "text", "text": content, "marks": [{"type": "underline"}]}]

    def code(self, content):
        return [{"type": "text", "text": content, "marks": [{"type": "code"}]}]

    def link(self, content, href):
        return [
            {
                "type": "text",
                "text": content,
                "marks": [
                    {"type": "link", "attrs": {"href": href, "target": "_blank"}}
                ],
            }
        ]
