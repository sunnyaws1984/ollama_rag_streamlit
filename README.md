# 🧠 RAG with Ollama and Streamlit

This project demonstrates a **Retrieval-Augmented Generation (RAG)** app using **Ollama** (local LLM), **FAISS** (vector store), and **Streamlit** (UI).  
It loads a **Machine Learning PDF from the web**, creates embeddings, retrieves relevant chunks, and generates grounded answers — all locally.

---

## 🚀 Features

- 🧩 Loads a Machine Learning research paper directly from the web  
- 🧮 Generates embeddings using **OllamaEmbeddings**  
- 🔍 Stores and retrieves chunks with **FAISS**  
- 💬 Answers user questions interactively in a **Streamlit UI**  
- ⚡ Caching for faster repeated queries  

---

## 🧰 Tech Stack

| Component | Description |
|------------|-------------|
| **Streamlit** | Frontend web UI |
| **Ollama** | Local LLM backend (e.g., LLaMA 3, Mistral, Phi 3) |
| **LangChain** | Framework for document loading, embeddings, and vector retrieval |
| **FAISS** | Vector similarity search |
| **Python 3.9+** | Core language |

---

## 📦 Installation

For Windows:
Go to the official download page:
👉 https://ollama.ai/download

Download the Windows installer (OllamaSetup.exe).
Run the installer — it will:
Install Ollama as a background service.
Add it automatically to your system PATH.
After installation, open a new terminal (PowerShell or Git Bash) and test:

ollama --version
ollama pull llama3
ollama list

1. **Clone this repo**
2 uv pip install -r requirements.txt or pip install -r requirements.txt
3. streamlit run rag_ollama_ml.py

## Concept
Document Loading → Downloads the Machine Learning paper from the web
Text Splitting → Breaks it into overlapping chunks
Embeddings → Uses OllamaEmbeddings to convert text → vectors
Vector Store → Stores embeddings in FAISS
Retrieval → Finds top relevant chunks for each query
Generation → LLM (via Ollama) generates a grounded answer
