import io
import uuid
import requests
from relevanceai.apps.report_app.marks import ReportMarks


class ReportBlocks(ReportMarks):
    def _process_content(self, content):
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        elif isinstance(content, list):
            content_list = []
            for c in content:
                if isinstance(c, str):
                    content_list.append({"type": "text", "text": c})
                elif isinstance(c, list):
                    content_list.append(c[0])
                else:
                    content_list.append(c)
            return content_list
        else:
            return content

    def h1(self, content, add=True):
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content": [
                {
                    "type": "heading",
                    "attrs": {"level": 1},
                    "content": self._process_content(content),
                }
            ],
        }
        if add:
            self.contents.append(block)
        return block

    def h2(self, content, add=True):
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content": [
                {
                    "type": "heading",
                    "attrs": {"level": 2},
                    "content": self._process_content(content),
                }
            ],
        }
        if add:
            self.contents.append(block)
        return block

    def h3(self, content, add=True):
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content": [
                {
                    "type": "heading",
                    "attrs": {"level": 3},
                    "content": self._process_content(content),
                }
            ],
        }
        if add:
            self.contents.append(block)
        return block

    def quote(self, content, add=True):
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content": [
                {"type": "blockquote", "content": self._process_content(content)}
            ],
        }
        if add:
            self.contents.append(block)
        return block

    def paragraph(self, content, add=True, raw=False):
        block = {"type": "paragraph", "content": self._process_content(content)}
        if raw:
            return block
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content": [block],
        }
        if add:
            self.contents.append(block)
        return block

    def space(self, height: int = 40, add=True):
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content": [
                {"type": "spaceBlock", "attrs": {"width": "100%", "height": height}}
            ],
        }
        if add:
            self.contents.append(block)
        return block

    def _list_item(self, content):
        return [{"type": "listItem", "content": [self.paragraph(content, raw=True)]}]

    def bullet_list(self, contents, add=True):
        if not isinstance(contents, list):
            raise TypeError("'contents' needs to be a List")
        list_contents = []
        for c in contents:
            list_contents += self._list_item(c)
        block = {
            "type": "appBlock",
            "content": [{"type": "bulletList", "content": list_contents}],
        }
        if add:
            self.contents.append(block)
        return block

    def ordered_list(self, contents, add=True):
        if not isinstance(contents, list):
            raise TypeError("'contents' needs to be a List")
        list_contents = []
        for c in contents:
            list_contents += self._list_item(c)
        block = {
            "type": "appBlock",
            "content": [{"type": "orderedList", "content": list_contents}],
        }
        if add:
            self.contents.append(block)
        return block

    def image(
        self, content, title: str = "", width_percentage: int = 100, add: bool = True
    ):
        if isinstance(content, str):
            if "http" in content and "/":
                # online image
                content_bytes = io.BytesIO(requests.get(content).content).getvalue()
            else:
                # local filepath
                content_bytes = io.BytesIO(open(content, "rb").read()).getvalue()
        elif isinstance(content, bytes):
            content_bytes = content
        elif isinstance(content, io.BytesIO):
            content_bytes = content.getvalue()
        else:
            raise TypeError("'content' needs to be of type str, bytes or io.BytesIO.")
        filename = f"{title}.png" if title else f"{str(uuid.uuid4())}.png"
        image_url = self.dataset.insert_media_bytes(
            content_bytes, filename=filename, verbose=False
        )
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content": [
                {
                    "type": "imageDisplayBlock",
                    "attrs": {
                        "imageSrc": image_url,
                        "title": title,
                        "width": f"{width_percentage}%",
                        "height": "auto",
                    },
                }
            ],
        }
        if add:
            self.contents.append(block)
        return block
