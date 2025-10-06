# RAG with Ollama and Streamlit

This project demonstrates a **Retrieval-Augmented Generation (RAG)** app using **Ollama** (local LLM), **FAISS** (vector store), and **Streamlit** (UI).  
It loads a **Machine Learning PDF from the web**, creates embeddings, retrieves relevant chunks, and generates grounded answers — all locally.

---

##  Features

- Load a Machine Learning research paper directly from the web  
- Generate embeddings using **OllamaEmbeddings** or **HuggingFaceEmbeddings**  
- Store and retrieve text chunks with **FAISS**  
- Answer user questions interactively in a **Streamlit UI**  
- Cache resources for faster repeated queries  

---

##  Tech Stack

| Component | Description |
|------------|-------------|
| **Streamlit** | Frontend web UI |
| **Ollama** | Local LLM backend (e.g., LLaMA 3, Mistral, Phi 3) |
| **LangChain** | Framework for document loading, embeddings, and vector retrieval |
| **FAISS** | Vector similarity search engine |
| **Python 3.9+** | Core programming language |

---

## Installation

### 1. Install Ollama

Go to the official download page: [Ollama Download](https://ollama.ai/download)  

For Windows:

1. Download the installer (`OllamaSetup.exe`)  
2. Run the installer (adds Ollama to system PATH)  
3. Open a new terminal and test:

```bash
ollama --version
ollama pull llama3
ollama list

git clone <repo-url>
cd <repo-folder>

pip install -r requirements.txt or uv pip install -r requirements.txt

streamlit run rag_ollama_ml.py