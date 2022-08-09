import io
import uuid
import requests
import numpy as np
from relevanceai.apps.report_app.marks import ReportMarks


class ReportBlocks(ReportMarks):
    def _process_content(self, content):
        # Needs to be done better
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        elif isinstance(content, (float, int, np.generic)):
            return [{"type": "text", "text": str(content)}]
        elif isinstance(content, list):
            content_list = []
            for c in content:
                if isinstance(c, list) and len(c) == 1:
                    content_list.append(c[0])
                else:
                    processed_content = self._process_content(c)
                    if not isinstance(processed_content, list):
                        processed_content = [processed_content]
                    content_list += processed_content
            return content_list
        elif isinstance(content, dict):
            return [content]
        else:
            print(type(content))
            return content

    def markdown(self):
        pass

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

    def code(self, content, language="python", add=True):
        block = {
            "type": "codeBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "attrs": {"language": language},
            "content": [
                {"type": "blockquote", "content": self._process_content(content)}
            ],
        }
        if add:
            self.contents.append(block)
        return block

    def paragraph(self, content, add=True, raw=False):
        p_block = {"type": "paragraph", "content": self._process_content(content)}
        if raw:
            return p_block
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content": [p_block],
        }
        if add:
            self.contents.append(block)
        return block

    p = paragraph

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

    def tooltip(self, content, tooltip_text, add=True):
        # has to be text block here.
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content": [
                {
                    "type": "tooltip",
                    "attrs": {"content": tooltip_text},
                    "content": self._process_content(content),
                }
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

    def details(self, title_content, contents, collapsed: bool = True, add=True):
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content": [
                {
                    "type": "details",
                    "attrs": {"open": collapsed},
                    "content": [
                        {
                            "type": "detailsSummary",
                            "content": self._process_content(title_content),
                        },
                        {
                            "type": "detailsContent",
                            "content": self._process_content(contents),
                        },
                    ],
                }
            ],
        }
        if add:
            self.contents.append(block)
        return block

    def card(self, contents, width, color, add=True):
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content": [
                {
                    "type": "cardBlock",
                    "attrs": {"width": width, "colour": color},
                    "content": self._process_content(contents),
                }
            ],
        }
        if add:
            self.contents.append(block)
        return block

    def _column_content(self, content):
        return [{"type": "columnContent", "content": self._process_content(content)}]

    def columns(self, contents, num_columns: int = 2, add=True):
        if not isinstance(contents, list):
            raise TypeError("'contents' needs to be a List")
        list_contents = []
        for c in contents:
            list_contents += self._column_content(c)
        block = {
            "type": "appBlock",
            "content": [
                {
                    "type": "columnBlock",
                    "attrs": {"columns": num_columns},
                    "content": list_contents,
                }
            ],
        }
        if add:
            self.contents.append(block)
        return block

    # def table(self, data, add=True):
    #     if data:
    #         table_headers = data.columns
    #         table_rows =
    #     block = {
    #         "type": "appBlock",
    #         "content": [{"type": "table", "content": table_rows}],
    #     }
    #     if add:
    #         self.contents.append(block)
    #     return block

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

    def audio(
        self, content, title: str = "", width_percentage: int = 100, add: bool = True
    ):
        if isinstance(content, str):
            if "http" in content and "/":
                # online audio
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
        filename = f"{title}.wav" if title else f"{str(uuid.uuid4())}.wav"
        audio_url = self.dataset.insert_media_bytes(
            content_bytes, filename=filename, verbose=False
        )
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content": [
                {
                    "type": "audioDisplayBlock",
                    "attrs": {
                        "audioSrc": audio_url,
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

    def video(
        self, content, title: str = "", width_percentage: int = 100, add: bool = True
    ):
        if isinstance(content, str):
            if "http" in content and "/":
                # online video
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
        filename = f"{title}.mp4" if title else f"{str(uuid.uuid4())}.mp4"
        video_url = self.dataset.insert_media_bytes(
            content_bytes, filename=filename, verbose=False
        )
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content": [
                {
                    "type": "videoDisplayBlock",
                    "attrs": {
                        "videoSrc": video_url,
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
