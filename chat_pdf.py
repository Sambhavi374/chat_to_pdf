from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

pdf_path = Path(__file__).parent / "Python_Handbook.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load() ## makes pages of the documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(documents) ## splits the documents into smaller chunks

embeddings = OpenAIEmbeddings()

