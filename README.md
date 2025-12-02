# 🧠 Local GPT RAG Assistant — Mini Project

A lightweight **Retrieval-Augmented Generation (RAG)** system that allows users to upload documents, index them locally, and ask questions **strictly based on the uploaded data**.

This project is built as part of the **AI Intern / Junior AI Developer Assessment**.

---

## 🚀 Overview

This application allows users to:

- Upload files: **PDF**, **TXT**, **CSV**, **DOCX**
- Convert the files into plain text
- Split the text into meaningful chunks (300–500 words)
- Generate embeddings using **SentenceTransformers**
- Store embeddings inside a local **FAISS** vector database
- Ask questions through a simple **Streamlit** UI
- Get answers from an LLM using only the retrieved chunks
- If no relevant content is found, the system responds:

  **“I don’t know based on the provided documents.”**

---

## 🧩 Architecture (RAG Flow)

```
Upload Files → Extract Text → Chunk Text → Embed Chunks → Store in FAISS
                                                                       ↓
User Query → Embed Query → Vector Search → Retrieve Top Chunks → LLM Generates Answer
                                                                       ↓
If relevance score is low → “I don’t know.”
```

---

## 📂 Project Structure

```
rag_app/
│── app.py                   # Streamlit UI
│── rag_engine/
│     ├── extractor.py       # File-to-text processing
│     ├── chunker.py         # Chunk creation
│     ├── embedder.py        # Embedding model & generation
│     ├── vector_store.py    # FAISS storage & search
│     ├── generator.py       # LLM-based answer generator
│── requirements.txt
│── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| Language | Python 3.10+ |
| UI | Streamlit |
| Embeddings | SentenceTransformers (all-mpnet-base-v2) |
| Vector Database | FAISS |
| LLM | OpenAI API / HuggingFace model |
| Document Parsing | PyPDF2, python-docx, pandas |

---

## 📄 Features

### ✔ File Upload & Text Extraction  
Supports **PDF**, **TXT**, **CSV**, and **DOCX** formats.

### ✔ Text Chunking  
Splits long documents into 300–500-word chunks to improve search granularity.

### ✔ Embedding Generation  
Creates dense vector embeddings using SentenceTransformers.

### ✔ Vector Search  
Uses FAISS to retrieve the most relevant document chunks.

### ✔ RAG Answer Generation  
The LLM responds strictly using retrieved context.  
If no useful context is found → responds:

**“I don’t know based on the provided documents.”**

### ✔ Simple & Fast Streamlit UI  
Upload → Process → Ask → Get Answer.

---

## 💡 Example Usage

### **Document Content**
> “Python is a high-level programming language created by Guido van Rossum.”

### **User Question**
> Who created Python?

### **Output**
> Python was created by Guido van Rossum.

### **Out-of-scope Example**
User: *Explain black holes.*  
System:  
> “I don’t know based on the provided documents.”

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone <your_repo_link>
cd rag_app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app
```bash
streamlit run app.py
```

---

## 🔐 Environment Variables (if using OpenAI)

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

---

## 📤 Deployment Options

This app can be deployed easily on:

- Streamlit Cloud
- Render
- HuggingFace Spaces
- LocalTunnel / Ngrok (temporary)

Ensure the link is **publicly accessible** for HR evaluation.

---

## 🧪 Handling Out-of-Scope Queries

The system:

1. Embeds the user’s question  
2. Compares similarity with stored document chunks  
3. If **relevance < threshold**, it returns the fallback message:

> “I don’t know based on the provided documents.”

This ensures **zero hallucination** and correct RAG behavior.

---

## 📘 Deliverables Included

- ✔ Fully functional RAG app  
- ✔ Modular and clean code  
- ✔ Live demo link  
- ✔ Public GitHub repository  
- ✔ This README.md  

---

## ⭐ Conclusion

This project demonstrates:

- Clear understanding of RAG architecture  
- Ability to build a complete document-question-answer pipeline  
- Clean modular code structure  
- Proper use of embeddings, vector search, and LLMs  
- UI integration with Streamlit  

---

