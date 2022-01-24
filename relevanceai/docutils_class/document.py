from doc_utils.doc_utils import DocUtils


class Document(DocUtils):
    def __init__(self, document) -> None:
        super().__init__()

        self.document = document

    def __getitem__(self, index):
        raise NotImplementedError

    def __setitem__(self, index, value):
        raise NotImplementedError
