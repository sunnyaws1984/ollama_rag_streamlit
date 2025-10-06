import streamlit as st
import requests
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Ollama LLM import
try:
    from langchain_ollama import Ollama
except ImportError:
    from langchain_community.llms import Ollama

# -------------------------------------
#  Configuration
# -------------------------------------
PDF_URL = "https://pop.princeton.edu/sites/g/files/toruqf496/files/documents/2021Jan_MachineLearning_0.pdf"
EMBED_MODEL = "all-MiniLM-L6-v2"  # Hugging Face embedding model
LLM_MODEL = "llama3"               # Ollama LLM for answers

st.set_page_config(page_title="RAG with HuggingFace + Ollama", layout="wide")
st.title("🤖 Machine Learning RAG App (HuggingFace + Ollama)")
st.write("Ask questions from the paper: [A Brief Introduction to Machine Learning for Engineers](https://pop.princeton.edu/sites/g/files/toruqf496/files/documents/2021Jan_MachineLearning_0.pdf)")

# -------------------------------------
# Step 1: Load PDF
# -------------------------------------
@st.cache_resource
def load_pdf_from_url(url):
    st.info("📥 Downloading PDF from the internet...")
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name
    loader = PyPDFLoader(tmp_path)
    return loader.load()

docs = load_pdf_from_url(PDF_URL)
st.success(f"✅ Loaded {len(docs)} pages from PDF.")

# -------------------------------------
# Step 2: Split into Chunks
# -------------------------------------
@st.cache_resource
def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)

chunks = create_chunks(docs)
st.write(f"📚 Split into {len(chunks)} text chunks.")

# -------------------------------------
# Step 3: Create / Load FAISS Index
# -------------------------------------
@st.cache_resource
def build_vectorstore(chunks):
    st.info("🔍 Creating embeddings using Hugging Face model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

# Try loading cached FAISS index first
try:
    st.info("🧠 Loading existing FAISS index...")
    vectorstore = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name=EMBED_MODEL))
    st.success("✅ Loaded FAISS index from disk.")
except Exception:
    st.warning("⚠️ No existing FAISS index found, creating a new one (this may take a few minutes)...")
    vectorstore = build_vectorstore(chunks)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------------------
# Step 4: User Query
# -------------------------------------
query = st.text_input("💡 Ask a question about the Machine Learning paper:")

if query:
    st.write("🔎 Searching relevant sections...")
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    # -------------------------------------
    # Step 5: Generate Answer via Ollama
    # -------------------------------------
    llm = Ollama(model=LLM_MODEL)
    prompt = f"""You are an expert assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {query}
Answer:"""

    with st.spinner("🧠 Generating answer..."):
        answer = llm.invoke(prompt)

    st.subheader("📝 Answer")
    st.write(answer)

    with st.expander("📄 Retrieved Context"):
        st.write(context)
else:
    st.info("👉 Type a question above to get started.")
