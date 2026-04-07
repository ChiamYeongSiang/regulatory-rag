import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ---- CONFIGURATION ----
import openai
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
PDF_FILES = ["FDA.pdf", "EUMDR.pdf"]

# ---- LOAD & PROCESS DOCUMENTS ----
@st.cache_resource
def load_rag_components():
    # Load PDFs
    all_pages = []
    for pdf in PDF_FILES:
        loader = PyPDFLoader(pdf)
        all_pages.extend(loader.load())

    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(all_pages)

    # Create vector database
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Set up retriever — increased to 6 chunks for better coverage
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # Set up LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Set up prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful regulatory affairs assistant specializing 
    in medical devices. Answer the question based ONLY on the 
    context provided below. If the answer is not in the context, 
    say "I don't have enough information in my documents to answer this."
    
    When comparing FDA and EU MDR, use information from both 
    documents if available.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    return retriever, llm, prompt

# ---- HELPER FUNCTIONS ----
def get_answer_with_sources(question, retriever, llm, prompt):
    # Get relevant chunks
    docs = retriever.invoke(question)
    
    # Format context
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Get sources
    sources = list(set([
        os.path.basename(doc.metadata.get('source', 'Unknown')) 
        for doc in docs
    ]))
    
    # Get answer
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return answer, sources

# ---- STREAMLIT UI ----
st.title("🤖 Regulatory AI Assistant")
st.caption("Ask questions about FDA and EU MDR regulations")

# ---- SIDEBAR ----
with st.sidebar:
    st.header("📚 Loaded Documents")
    for pdf in PDF_FILES:
        st.success(f"✅ {pdf}")
    
    st.divider()
    st.header("ℹ️ About")
    st.write("This assistant answers questions based on:")
    st.write("• FDA Medical Device Guidance (2026)")
    st.write("• EU MDR Compliance Guide")
    
    st.divider()
    st.caption("Built with LangChain + ChromaDB + OpenAI")

# ---- LOAD RAG COMPONENTS ----
with st.spinner("Loading documents... please wait..."):
    retriever, llm, prompt = load_rag_components()

st.success("✅ Documents loaded! Ask me anything.")

# ---- CHAT HISTORY ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            st.caption(f"📄 Sources: {', '.join(message['sources'])}")

# ---- CHAT INPUT ----
if question := st.chat_input("Ask a question about FDA or EU MDR..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get RAG answer with sources
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer, sources = get_answer_with_sources(
                question, retriever, llm, prompt
            )
        st.markdown(answer)
        st.caption(f"📄 Sources: {', '.join(sources)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })