import streamlit as st
from groq import Groq
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from retriever import get_relevant_chunks

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="Document Q&A", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None:
        if st.session_state.doc_name != uploaded_file.name:
            with st.spinner("Processing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                documents = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = splitter.split_documents(documents)

                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
                vectorstore = FAISS.from_documents(chunks, embeddings)
                vectorstore.save_local("faiss_index")

                os.unlink(tmp_path)

                st.session_state.doc_name = uploaded_file.name
                st.session_state.chunk_count = len(chunks)
                st.session_state.index_ready = True
                st.session_state.messages = []

    if st.session_state.index_ready:
        st.success("Document loaded")
        st.write("File:", st.session_state.doc_name)
        st.write("Chunks indexed:", st.session_state.chunk_count)

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("Upload a PDF to get started.")

st.title("Document Q&A")

if not st.session_state.index_ready:
    st.write("Upload a PDF using the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                chunks = get_relevant_chunks(prompt)

            system_prompt = f"""You are a helpful assistant that answers questions
based on the provided document context. If the answer is not in the context,
say so clearly.

Context from document:
{chunks}"""

            with st.spinner("Generating response..."):
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *[{"role": m["role"], "content": m["content"]}
                          for m in st.session_state.messages]
                    ]
                )
                answer = response.choices[0].message.content

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})