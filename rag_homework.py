"""# 1. Importing Libraries"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

"""# 2. Loading the pdf"""

loader = PyPDFLoader("RAG.pdf")
documents = loader.load()

"""# 3. Chunking the document"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=2
)

chunks = splitter.split_documents(documents)

print(chunks)
print(len(chunks))

"""# 4. Embedding Model"""

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

"""# 5. Storing the vector in Vector DB"""

db = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

"""# 6. Similarity Search"""

query = "What are the three main stages of a RAG pipeline?"

results = db.similarity_search(query)

print(results[0].page_content)

