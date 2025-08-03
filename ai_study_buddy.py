# AI Study Buddy - PDF Q&A Chatbot with Streamlit + LangChain

import streamlit as st
import fitz  # PyMuPDF
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import os

# Set your OpenAI API key (or use Streamlit secrets)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="📚 AI Study Buddy", layout="wide")
st.title("📚 AI Study Buddy")
st.markdown("Ask questions from your uploaded PDF notes using GPT-4 + vector search.")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Reading and processing your PDF..."):
        # Step 1: Extract text from PDF
        pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        raw_text = ""
        for page in pdf_reader:
            raw_text += page.get_text()

        # Step 2: Chunk the text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = splitter.split_text(raw_text)
        documents = [Document(page_content=t) for t in texts]

        # Step 3: Create vector store with FAISS
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Step 4: Set up RetrievalQA chain
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

    st.success("PDF processed! Ask your questions below.")
    user_question = st.text_input("💬 Ask a question from the PDF")

    if user_question:
        with st.spinner("Thinking..."):
            result = qa_chain({"query": user_question})
            st.subheader("Answer:")
            st.write(result["result"])

            with st.expander("🔍 Source Chunks"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)

else:
    st.info("Please upload a PDF file to get started.")
