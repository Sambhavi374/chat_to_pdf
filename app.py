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

LLM = ChatGroq(groq_api_key=groq_api_key, 
            model="llama3-8b-8192", 
            temperature=0.0)

# App title and description
st.set_page_config(page_title="Document QnA", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Set the entire background color */
    body {
        background-color: 	#e0bbf7;
         color: #000000; /* Black text for all elements */
    }
    /* Main title styling */
    .main-title {
        font-size: 36px;
        color: #000000; /* Black text */
        text-align: center;
        font-weight: bold;
    }
    /* Sub-title styling */
    .sub-title {
        font-size: 20px;
        color: #000000; /* Black text */
        margin-top: 20px;
    }
    /* Success message styling */
    .success-message {
        color: #2c3e50; 
        font-weight: bold;
    }
    /* Warning message styling */
    .warning-message {
        color: #2c3e50;
        font-weight: bold;
    }
    /* Response box styling */
    .response-box {
        background-color: rgba(173, 216, 230, 0.3); /* Light blue with 0.3 opacity */
        padding: 10px;
        border-radius: 5px;
        border: 1px solid ##dbb2ff; 
        color: #000000;
    }
    /* Expander title styling */
    .expander-title {
        color: #000000; /* Black text */
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main title
st.markdown('<div class="main-title">📄 Document QnA</div>', unsafe_allow_html=True)

# Sub-title
st.markdown(
    """
    <div class="sub-title">
    Welcome to <b>Document QnA</b>! This app allows you to upload PDF files, create embeddings, and ask questions based on the document's content.
    </div>
    """,
    unsafe_allow_html=True,
)


# Main content
st.markdown('<div class="sub-title">🔧 Create Embeddings</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="warning-message">
    Before asking a question, please create the embeddings by clicking the button below.<br>
    <b>Note:</b> For questions from the same document, embeddings need to be created only once.
    </div>
    """,
    unsafe_allow_html=True,
)


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

# Sidebar for file upload
st.sidebar.header("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    # Save the uploaded file to the 'data' folder
    data_folder = "./data"
    os.makedirs(data_folder, exist_ok=True)
    file_path = os.path.join(data_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully!")



if st.button("🚀 Create Embeddings"):
    vec_embedding()
    st.markdown('<div class="success-message">Vector store created successfully!</div>', unsafe_allow_html=True)

# Question input
st.markdown('<div class="sub-title">❓ Ask a Question</div>', unsafe_allow_html=True)
prompt1 = st.text_input("Type your question here:")


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
    
    st.markdown('<div class="sub-title">📝 Response</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)

    with st.expander("🔍 Document Similarity Search"):
        st.markdown('<div class="expander-title">Similar Documents:</div>', unsafe_allow_html=True)
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Document {i + 1}:**")
            st.write(doc.page_content)
            st.write("-------------------------------------------")