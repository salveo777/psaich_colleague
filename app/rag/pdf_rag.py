from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # use FAISS for loval efficiency
from langchain_community.embeddings import HuggingFaceBgeEmbeddings # better tested than OllamaEmbeddings
# from langchain_community.chains import RetrievalQA  
from langchain_core.documents import Document
import os
from typing import List

class SimplePDFRAG:
    def __init__(self, pdf_path: str = "data/knowledge_folder", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.pdf_path = pdf_path
        self.embedding_model = HuggingFaceBgeEmbeddings(model_name=embedding_model)
        self.vector_store = None
        self.retriever = None
        self._build_vector_store()
        
    def _load_pdfs(self) -> List[Document]:
        documents = []
        print(f"Search PDFs in: {os.path.abspath(self.pdf_path)}")
        all_files = os.listdir(self.pdf_path)
        print("Found files:", all_files)
        for filename in all_files:
            if filename.lower().endswith(".pdf"):
                full_path = os.path.join(self.pdf_path, filename)
                print(f"Load PDF: {full_path}")
                try:
                    loader = PyPDFLoader(full_path)
                    docs = loader.load()
                    print(f"  -> {len(docs)} document(s) loaded from {filename}")
                    for doc in docs:
                        doc.metadata['source'] = filename
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        print(f"Total documents loaded: {len(documents)}")
        return documents
        
    def _build_vector_store(self):
        documents = self._load_pdfs()
        if not documents:
            self.vector_store = None
            self.retriever = None
            return
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
        chunkgs = text_splitter.split_documents(documents)
        # Create a vector store from the documents
        self.vector_store = FAISS.from_documents(chunkgs, self.embedding_model)
        self.retriever = self.vector_store.as_retriever()

    def retrieve(self, query: str) -> List[str]:
        if not self.retriever:
            return []
        docs = self.retriever.get_relevant_documents(query)
        print("Retrieved docs from sources:", list(set(doc.metadata.get('source', '') for doc in docs)))
        return [f"[{doc.metadata.get('source','')}] {doc.page_content}" for doc in docs]
    
    def reload(self):
        """Reload the vector store from the PDF files."""
        self._build_vector_store()
