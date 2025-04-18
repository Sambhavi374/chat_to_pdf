from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore


import getpass
import os

pdf_path = Path(__file__).parent / "Python_Handbook.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load() ## makes pages of the documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(documents) ## splits the documents into smaller chunks


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("AIzaSyA6bU0-anJWB8ZKb-kNAR8ebYEentwvoGU")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

vector_store = QdrantVectorStore.from_documents(
    documents = [],
    url = "http://localhost:6333",
    collection_name = "pdf",
    embedding = embeddings,
)

vector_store.add_documents(split_docs) ## adds the split documents to the vector store
print("completed data ingestion")

retriver = QdrantVectorStore.from_existing_collection(
    url = "http://localhost:6333",
    collection_name = "pdf",
    embedding = embeddings,
)

search_results = retriver.similarity_search(
    query = "What is the purpose of the __init__ method in Python?", 
    embedding = embeddings
)

print("Search results:", search_results)
