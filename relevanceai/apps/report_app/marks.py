import warnings
from relevanceai.apps.report_app.base import ReportBase


class ReportMarks(ReportBase):
    def _process_marks(self):
        return

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

    def highlight(self, content, start, end, color):
        return [
            {
                "type": "text",
                "text": content,
                "marks": [
                    {
                        "type": "highlight",
                        "attrs": {"color": color},
                        "from": start,
                        "to": end,
                    }
                ],
            }
        ]

    def color(self, content, color, background_color):
        return [
            {
                "type": "text",
                "text": content,
                "marks": [
                    {
                        "type": "textStyle",
                        "attrs": {"color": color, "backgroundColor": background_color},
                    }
                ],
            }
        ]

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
