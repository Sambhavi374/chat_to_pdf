import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS # vector store
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings # vector embedding technique
import time 

from dotenv import load_dotenv

load_dotenv()

## load GROQ and GOOGLE API keys

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Document QnA")

LLM = ChatGroq(groq_api_key=groq_api_key, 
            model="llama3-8b-8192", 
            temperature=0.0)


prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the context provided only.
    Please provide the most accurate response based on the question.
    If the question requires a detailed answer, please provide a detailed response.
    If the question requires a short answer, please provide a short response.
    If the question is not related to the context, please respond with "Question out of context".

    Context: {context}
    Question: {question}
    Answer:
    """
)

def vec_embedding():


    if "vectors" not in st.session_state:
        st.session_state.embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        st.session_state.loader = PyPDFDirectoryLoader("./data") #data ingestion
        st.session_state.docs = st.session_state.loader.load() ## loads documents

        ## chunking the documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        ## creating vector store
        st.session_state.vectors= FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embedding
        )

# File uploader for user to upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    # Save the uploaded file to the 'data' folder
    data_folder = "./data"
    os.makedirs(data_folder, exist_ok=True)
    file_path = os.path.join(data_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")        

prompt1=st.text_input("Ask your Question here")

st.markdown("_Before asking a question, please create the embeddings by clicking the button below._")
st.markdown("*Note:*_For the question from same document, embeddings need to to created once only._")

if st.button("Create Embeddings"):
    vec_embedding()
    st.write("Vector store created successfully!")


if prompt1:
    document_chain = create_stuff_documents_chain(LLM, prompt=prompt)
    retriever = st.session_state.vectors.as_retriever()

    start = time.process_time()

    # Step 1: Get context documents
    retrieved_docs = retriever.invoke(prompt1)

    # Step 2: Now use document_chain to send both context and question
    response = document_chain.invoke({
        "context": retrieved_docs,
        "question": prompt1
    })

    st.write(response)

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(retrieved_docs):
            st.write(doc.page_content)
            st.write("-------------------------------------------")