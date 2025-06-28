from app.communicator.communicator import Communicator


def test_pdf_rag_loading(self):
    """Test if PDFs are loaded and indexed by SimplePDFRAG."""
    if self.pdf_rag and self.pdf_rag.vector_store:
        print(f"PDF RAG loaded. Number of documents: {len(self.pdf_rag.vector_store.docstore._dict)}")
    else:
        print("PDF RAG not loaded or no documents found.")


comm = Communicator()
list(comm.pdf_rag.vector_store.docstore._dict.keys())[10:15]
comm.pdf_rag.vector_store.docstore._dict['06c295fd-aa54-45a4-a9fc-f36abeb6d0f8']