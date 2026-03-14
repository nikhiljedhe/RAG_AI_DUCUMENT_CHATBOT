
import streamlit as st
import os
from document_loader import load_document, chunk_documents
from vector_store import build_vector_store, get_retriever
from rag_pipeline import build_qa_chain, query, format_sources

os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

st.set_page_config(page_title="DocChat AI", page_icon="📄", layout="wide")

with st.sidebar:
    st.title("⚙️ Settings")
    model_choice = st.selectbox("Groq Model", [
        "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "llama-4-scout-17b-16e-instruct", "llama-4-maverick-17b-128e-instruct",
         "qwen/qwen3-32b",
    ])
    chunk_size = st.slider("Chunk Size", 300, 2000, 1000, 100)
    top_k = st.slider("Top-K Chunks", 1, 8, 4)
    st.divider()
    st.markdown("**Free Stack:**")
    st.markdown("- 🤖 LLM: Groq (LLaMA 3)")
    st.markdown("- 🧠 Embeddings: HuggingFace")
    st.markdown("- 🗂️ Vector DB: FAISS")
    if st.button("🗑️ Clear Chat & Reset", use_container_width=True):
        for k in ["qa_chain", "messages", "doc_name", "num_chunks"]:
            st.session_state.pop(k, None)
        st.rerun()

st.title("📄 AI Document Q&A Chatbot")
st.caption("Upload a PDF or TXT and ask questions — 100% free, no OpenAI key needed.")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

if uploaded_file and (
    "qa_chain" not in st.session_state
    or st.session_state.get("doc_name") != uploaded_file.name
    or st.session_state.get("model") != model_choice
):
    with st.status("⚙️ Processing document...", expanded=True) as status:
        st.write("📂 Loading...")
        docs = load_document(uploaded_file)
        st.write("✂️ Chunking...")
        chunks = chunk_documents(docs, chunk_size=chunk_size)
        st.write("🧠 Generating embeddings (first run downloads ~90MB model)...")
        vs = build_vector_store(chunks)
        st.write(f"🔗 Building RAG chain with {model_choice}...")
        retriever = get_retriever(vs, k=top_k)
        st.session_state.qa_chain = build_qa_chain(retriever, model_name=model_choice)
        st.session_state.messages = []
        st.session_state.doc_name = uploaded_file.name
        st.session_state.num_chunks = len(chunks)
        st.session_state.model = model_choice
        status.update(label="✅ Ready!", state="complete")
    st.success(f"Indexed **{len(chunks)} chunks** from *{uploaded_file.name}*")

if "qa_chain" in st.session_state:
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Document", st.session_state.doc_name)
    c2.metric("Chunks", st.session_state.num_chunks)
    c3.metric("Model", st.session_state.model)
    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Sources"):
                    st.markdown(msg["sources"])

    if prompt := st.chat_input("Ask a question about your document..."):
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, source_docs = query(st.session_state.qa_chain, prompt)
                sources_text = format_sources(source_docs)
            st.write(answer)
            if sources_text:
                with st.expander("📚 Sources"):
                    st.markdown(sources_text)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_text})
else:
    st.info("👆 Upload a PDF or TXT file above to get started.")
