
class ReportMarks:
    def bold(self, content):
        return [{
            "type":"text",
            "text": content,
            "marks":[{"type":"bold"}]
        }]

    def italic(self, content):
        return [{
            "type":"text",
            "text": content,
            "marks":[{"type":"italic"}]
        }]

    def strike(self, content):
        return [{
            "type":"text",
            "text": content,
            "marks":[{"type":"strike"}]
        }]

    def code(self, content):
        return [{
            "type":"text",
            "text": content,
            "marks":[{"type":"code"}]
        }]

    def link(self, content, href):
        return [{
            "type":"text",
            "text": content,
            "marks":[{"type":"link", "attrs" : {"href" : href, "target":"_blank"}}]
        }]

class ReportApp(ReportMarks):
    def __init__(self, name):
        self.name = name
        self.app = []

    def _process_content(self, content):
        if isinstance(content, str):
            return [{"type":"text", "text":content}]
        else:
            return content

    def h1(self, content, add=True):
        block = {
            "type" : "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content" : [{
                "type": "heading", 
                "attrs" : {"level" : 1},
                "content" : self._process_content(content)
            }]
        }
        if add: self.app.append(block)
        return block
    
    def h2(self, content, add=True):
        block = {
            "type" : "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content" : [{
                "type": "heading", 
                "attrs" : {"level" : 2},
                "content" : self._process_content(content)
            }]
        }
        if add: self.app.append(block)
        return block

    def h3(self, content, add=True):
        block = {
            "type" : "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content" : [{
                "type": "heading", 
                "attrs" : {"level" : 3},
                "content" : self._process_content(content)
            }]
        }
        if add: self.app.append(block)
        return block

    def quote(self, content, add=True):
        block = {
            "type" : "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content" : [{
                "type": "blockquote", 
                "content" : self._process_content(content)
            }]
        }
        if add: self.app.append(block)
        return block

    def paragraph(self, content, add=True, raw=False):
        block = {
            "type": "paragraph", 
            "content" : self._process_content(content)
        }
        if raw: return block
        block = {
            "type" : "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content" : [block]
        }
        if add: self.app.append(block)
        return block

    def _list_item(self, content):
        return [{
            "type" : "listItem",
            "content" : [self.paragraph(content, raw=True)]
        }]

    def bullet_list(self, contents, add=True):
        if not isinstance(contents, list):
            raise TypeError("'contents' needs to be a List")
        list_contents = []
        for c in contents:
            list_contents += self._list_item(c)
        block = {
            "type" : "appBlock",
            "content" : [{
                "type": "bulletList", 
                "content" : list_contents
            }]
        }
        if add: self.app.append(block)
        return block


    def ordered_list(self, contents, add=True):
        if not isinstance(contents, list):
            raise TypeError("'contents' needs to be a List")
        list_contents = []
        for c in contents:
            list_contents += self._list_item(c)
        block = {
            "type" : "appBlock",
            "content" : [{
                "type": "orderedList", 
                "content" : list_contents
            }]
        }
        if add: self.app.append(block)
        return block
