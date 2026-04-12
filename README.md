# 🤖 Medical Device Regulatory AI Assistant

A RAG-based AI chatbot that enables natural language querying 
of medical device regulatory documents — built with LangChain, 
FAISS, OpenAI, and Streamlit.

🔗 **Live App:** https://regulatory-rag.streamlit.app

---

## What it does

Regulatory professionals spend hours manually searching through 
FDA, EU MDR, ISO 13485 and other compliance documents to answer 
questions. This app solves that problem by letting users ask 
questions in plain English and get instant, sourced answers.

---

## Tech stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| Vector store | FAISS |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | GPT-3.5-turbo |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

---

## Documents loaded

- EU MDR Compliance Guide
- ISO 13485:2016 Quality Management
- FDA 21 CFR Part 11 Guidance
- FDA Process Validation Guidance
- Medical Device Quality System Amendment

---

## Sample questions to try

- What are the key requirements of ISO 13485?
- What are the three stages of process validation according to FDA?
- What are the General Safety and Performance Requirements under EU MDR?
- Tell me the difference between EU MDR and ISO 13485
- What are the requirements for electronic records under 21 CFR Part 11?

---

## How it works

1. PDFs are loaded and split into 500 character chunks
2. Each chunk is converted to a vector using OpenAI embeddings
3. Vectors are stored in a FAISS index
4. User question is embedded and matched against stored vectors
5. Top 10 most relevant chunks are retrieved
6. GPT-3.5-turbo generates an answer based on retrieved context
7. Sources are displayed with every answer

---

## Roadmap

**Version 1.0 (Current) ✅**
- 5 regulatory documents
- Natural language querying
- Source citations
- Comparative query support

**Version 1.1 (Planned)**
- User feedback button
- Additional regulatory documents
- Improved hallucination detection

**Version 2.0 (Future)**
- User document upload
- Multi language support
- Agent that compares specific regulatory clauses

---

## Built by

Sophia Yeong Siang Chiam
- 10 years in medical device and regulated industries
- MS Business Analytics, University of Arizona (2026)
- LinkedIn: https://www.linkedin.com/in/sophia-chiam-yeong-siang/
