from doc_utils.doc_utils import DocUtils


class DocumentList(DocUtils):
    def __init__(self, documents) -> None:
        super().__init__()

        self.documents = documents

    def __getitem__(self, index):
        raise NotImplementedError

    def __setitem__(self, index, value):
        raise NotImplementedError
