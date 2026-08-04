# UnTangle 📄

> **Make sense of your complex documents.**
> UnTangle is an intelligent document assistant designed to turn dense, confusing files into clear, actionable insights. Simply upload any jargon-heavy document—like a legal contract, research paper, or corporate report—and chat directly with it to get instant answers, summaries, and clarity. 

---

## 📖 About UnTangle

Stop wasting hours deciphering dense text and confusing jargon. UnTangle is your personal AI reading assistant, built to help you instantly understand any document you throw at it. 

Whether you are a student tackling heavy academic papers, a professional reviewing fifty-page contracts, or just someone trying to understand a complicated user manual, UnTangle does the heavy lifting. Just upload your PDF or text file and start asking questions. UnTangle scans the contents in seconds to provide precise answers, plain-language summaries, and clarity on your specific doubts. 

## 🛠️ Built With

UnTangle is built with a privacy-first, local-LLM architecture:
* **[Streamlit](https://streamlit.io/):** For the interactive web user interface.
* **[LangChain](https://www.langchain.com/):** For orchestrating the document loading, splitting, and RAG (Retrieval-Augmented Generation) pipeline.
* **[Ollama](https://ollama.ai/):** For running large language models and embeddings entirely locally.
* **[Chroma](https://www.trychroma.com/):** For local vector database storage and semantic search.

## ✨ Key Features

* **Instant Document Upload:** Seamlessly upload PDFs and text files with a single click.
* **Conversational Q&A:** Ask questions in plain English and get immediate, context-aware answers pulled directly from your text.
* **100% Local & Private:** Documents never leave your machine thanks to local embedding and LLM processing via Ollama.
* **Jargon Translation:** Highlight complex terminology or dense paragraphs and ask UnTangle to explain it to you like a beginner.
* **Source Citations:** UnTangle always points you exactly to the page and paragraph where it found the answer, ensuring total accuracy.

## 🚀 Getting Started

### Prerequisites

To run this project, you will need the following installed on your machine:
* [Python 3.9 or higher](https://www.python.org/downloads/)
* [Ollama](https://ollama.ai/download) (You must have Ollama running locally)

You will also need to pull an Ollama model for the chat and embeddings. For example:
```bash
ollama pull llama3
ollama pull nomic-embed-text