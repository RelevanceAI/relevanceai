import warnings
from relevanceai.apps.report_app.base import ReportBase


class ReportMarks(ReportBase):
    def _process_marks(self, content, mark, nested=False):
        #Needs to be done better
        if isinstance(content, list):
            new_content = []
            for c in content:
                if isinstance(c, list) and len(c) == 1:
                    new_content.append(self._process_marks(c[0], mark, nested=True))
                else:
                    new_content.append(self._process_marks(c, mark, nested=True))
            return new_content
        elif isinstance(content, dict):
            if "marks" in content:
                content['marks'] += mark
            return content
        elif nested:
            return {"type": "text", "text": content, "marks": mark}
        else:
            return [{"type": "text", "text": content, "marks": mark}]

    def bold(self, content):
        return self._process_marks(content, [{"type": "bold"}])

    def italic(self, content):
        return self._process_marks(content, [{"type": "italic"}])

    def strike(self, content):
        return self._process_marks(content, [{"type": "strike"}])

    def underline(self, content):
        return self._process_marks(content, [{"type": "underline"}])

    def inline_code(self, content):
        return self._process_marks(content, [{"type": "code"}])

    def highlight(self, content, start, end, color):
        return self._process_marks(content, [
            {
                "type": "highlight",
                "attrs": {"color": color},
                "from": start,
                "to": end,
            }
        ])

    def color(self, content, color, background_color):
        return self._process_marks(content, [
            {
                "type": "textStyle",
                "attrs": {"color": color, "backgroundColor": background_color},
            }
        ])

    def link(self, content, href):
        return self._process_marks(content, [
            {"type": "link", "attrs": {"href": href, "target": "_blank"}}
        ])
