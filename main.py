"""
main.py
Simple RAG prototype for Assignment 1 (Kalpit - AI Intern)
- Loads speech.txt
- Splits into chunks
- Builds embeddings (HuggingFaceEmbeddings)
- Stores/reuses a Chroma vectorstore
- Runs a retrieval-augmented QA loop using Ollama (Mistral 7B)
"""

import os
from pathlib import Path
import argparse

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# LCEL
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# -----------------------
# Basic config
# -----------------------
BASE_DIR = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "corpus"
SPEECH_FILE = CORPUS_DIR / "speech.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------
# Utilities
# -----------------------
def ensure_dirs():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------
# Build or load vectorstore
# -----------------------
def build_vectorstore(force_rebuild: bool = False, chunk_size: int = 500, chunk_overlap: int = 50):
    ensure_dirs()

    if not SPEECH_FILE.exists():
        raise FileNotFoundError(f"Missing {SPEECH_FILE}. Please create it and paste the speech text.")

    loader = TextLoader(str(SPEECH_FILE), encoding="utf-8")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    splitted_docs = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if force_rebuild and CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)

    vectordb = Chroma.from_documents(
        documents=splitted_docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vectordb.persist()
    return vectordb


# -----------------------
# Build RAG chain (modern LCEL)
# -----------------------
def build_rag_chain(vectordb):
    llm = Ollama(model="qwen2.5:1.5b")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # ✨ Proper prompt so Ollama receives a string
    template = """
You are a helpful assistant. Use ONLY the given context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    rag_chain = (
        RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
    )

    return rag_chain


# -----------------------
# CLI Loop
# -----------------------
def interactive_loop(rag_chain):
    print("\nRAG Prototype - Ask questions about the speech (type 'exit' or 'quit' to stop)\n")

    while True:
        q = input("Q: ").strip()
        if q.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if q == "":
            continue

        try:
            answer = rag_chain.invoke(q)
        except Exception as e:
            print("Error querying LLM:", e)
            continue

        print("\nA:", answer)
        print("\n" + ("-" * 50) + "\n")


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Simple RAG CLI prototype for Ambedkar speech")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the Chroma vectorstore")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    print("Building / loading vectorstore...")
    vectordb = build_vectorstore(
        force_rebuild=args.rebuild,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print("Vectorstore ready. Building RAG chain...")
    rag_chain = build_rag_chain(vectordb)

    print("Ready. Start asking questions.")
    interactive_loop(rag_chain)


if __name__ == "__main__":
    main()
