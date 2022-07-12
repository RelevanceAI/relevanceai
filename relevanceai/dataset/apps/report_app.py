class ReportApp:
    def __init__(self, name):
        self.name = name
        self.app = []

    def _process_content(self, content):
        if isinstance(content, str):
            return [{"type":"text", "text":content}]
        else:
            return content

    def bold(self, content, add=False):
        return [{
            "type":"text",
            "text": content,
            "marks":[{"type":"bold"}]
        }]

    def italic(self, content, add=False):
        return [{
            "type":"text",
            "text": content,
            "marks":[{"type":"italic"}]
        }]

    def h1(self, content, add=False):
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
    
    def h2(self, content, add=False):
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

    def quote(self, content, add=False):
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

    def paragraph(self, content, add=False, raw=False):
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

    def bullet_list(self, contents, add=False):
        if not isinstance(contents, list):
            raise TypeError("'contents' needs to be a List")
        list_contents = []
        for c in contents:
            list_contents += [{
                "type" : "listItem",
                "content" : [self.paragraph(c, raw=True)]
            }]
        block = {
            "type" : "appBlock",
            "content" : [{
                "type": "bulletList", 
                "content" : list_contents
            }]
        }
        if add: self.app.append(block)
        return block