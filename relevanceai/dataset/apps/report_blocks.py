class ReportApp:
    def __init__(self, name):
        self.name = name
        self.app = []

    def _process_content(self, content):
        if isinstance(content, str):
            return [{"type":"text", "text":content}]
        else:
            return content

    def h1(self, content, add=False):
        block = {
            "type" : "appBlock",
            "content" : [{
                "type": "heading", 
                "attrs" : {"level" : 1},
                "content" : self._process_content(content)
            }]
        }
        if add:
            self.app.append(block)
        return block
    
    def h2(self, content, add=False):
        block = {
            "type" : "appBlock",
            "content" : [{
                "type": "heading", 
                "attrs" : {"level" : 2},
                "content" : self._process_content(content)
            }]
        }
        if add:
            self.app.append(block)
        return block

    def quote(self, content, add=False):
        block = {
            "type" : "appBlock",
            "content" : [{
                "type": "blockquote", 
                "content" : self._process_content(content)
            }]
        }
        if add:
            self.app.append(block)
        return block

    def bold(self, content, add=False):
        block = {
            "type":"text",
            "text": content,
            "marks":"bold"
        }
