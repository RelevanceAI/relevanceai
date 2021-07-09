"""Helper utilities
"""
class HelperMixin:
    def chunk(self, docs: list, chunksize: int=15):
        """Helper Mixin
        """
        for i in range(int(len(docs) / chunksize)):
            yield docs[i: (i+chunksize)]
    
    def show_pdf(self, pdf_filename: str, page_number: int):
        """Show PDF inside a Jupyter Notebook.
        """
        from pdf2image import convert_from_bytes
        return convert_from_bytes(
            open(pdf_filename, 'rb').read(),
            first_page=page_number,
            last_page=page_number + 1)[0]
