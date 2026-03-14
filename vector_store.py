
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_vector_store(chunks):
    return FAISS.from_documents(chunks, get_embeddings())

def get_retriever(vector_store, k=4):
    return vector_store.as_retriever(search_kwargs={"k": k})
