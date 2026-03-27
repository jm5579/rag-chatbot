import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
from retriever import get_relevant_chunks

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("Document Q&A")
st.caption("Ask questions about your uploaded document.")

if "messages" not in st.session_state:
    st.session_state.messages = []

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