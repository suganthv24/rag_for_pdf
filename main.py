from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = Path(__file__).with_name("RAG.pdf")


def build_vector_store(pdf_path: str = str(PDF_PATH)) -> FAISS:
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=2)
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No text could be extracted from the document. "
            "If this is a scanned PDF, OCR is required before embedding."
        )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(documents=chunks, embedding=embedding_model)


def search_documents(query: str, pdf_path: str = str(PDF_PATH), top_k: int = 3):
    db = build_vector_store(pdf_path)
    return db.similarity_search(query, k=top_k)



if __name__ == "__main__":
    query = "What are the three main stages of a RAG pipeline?"
    results = search_documents(query)

    if results:
        print(results[0].page_content)
    else:
        print("No relevant passages found.")

