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
