Assignment 1 RAG System (Kalpit AI Internship)

This project implements a Retrieval-Augmented Generation (RAG) pipeline
using LangChain, ChromaDB, HuggingFace embeddings, and Ollama (Mistral).
The system loads the provided speech.txt document, creates embeddings,
stores them in a vector database, and enables interactive
question-answering over the content.

  ------------------------------------------------------------
  FEATURES
  ------------------------------------------------------------
  - Loads and processes speech.txt - Splits text into
  overlapping chunks - Generates embeddings using
  all-MiniLM-L6-v2 - Stores vectors in a local Chroma
  database - Uses Ollama (Mistral) for LLM responses -
  Interactive question-answering loop - Uses modern LCEL
  (LangChain Expression Language) instead of deprecated
  RetrievalQA - Clean, modular code structure

  ------------------------------------------------------------

TECH STACK

Embeddings: HuggingFace all-MiniLM-L6-v2 Vector Store: ChromaDB LLM:
Ollama (Mistral 7B) Framework: LangChain Community + LCEL Language:
Python 3.10

  ------------------------------------------------------------
  PROJECT STRUCTURE
  ------------------------------------------------------------
  assignment-1/ 
├── corpus/ 
│ └── speech.txt 
├──  chroma_db/ 
├── main.py 
└── README.md

  ------------------------------------------------------------

INSTALLATION AND SETUP

1.  Create Python 3.10 virtual environment: python3.10 -m venv venv venv

2.  Install dependencies: pip install langchain langchain-community
    langchain-text-splitters langchain-huggingface sentence-transformers
    chromadb

3.  Install Ollama (Download from ollama.com)

4.  Pull the Mistral model: ollama pull mistral

  -------------------------
  RUNNING THE APPLICATION
  -------------------------

Activate environment and run:

python main.py

You will see:

Ready. Start asking questions. Q:

Ask any question about the speech.

Type “exit” to quit.

  ------------------------------------------------------------
  HOW IT WORKS
  ------------------------------------------------------------
  1. Loads speech.txt 2. Splits text into 500-character chunks
  3. Converts each chunk into vector embeddings 4. Stores
  vectors in ChromaDB 5. Retrieves the most relevant chunks
  for each question 6. Sends the context + question to the
  Mistral model 7. Outputs the final answer

  ------------------------------------------------------------
  EXAMPLE USAGE
  ------------------------------------------------------------
  Q: Who wrote the speech? A: Dr. B. R. Ambedkar…

  ------------------------------------------------------------

ASSIGNMENT STATUS

Assignment 1 completed successfully.


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/dc015413-d490-4bef-8d39-87ed67589739" />

