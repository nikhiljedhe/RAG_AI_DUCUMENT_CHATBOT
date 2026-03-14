
import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from prompts import QA_PROMPT

def build_qa_chain(retriever, model_name="llama-3.1-8b-instant"):
    llm = ChatGroq(
        model_name=model_name,
        temperature=0,
        max_tokens=600,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

def query(qa_chain, question):
    result = qa_chain.invoke({"query": question})
    return result["result"], result["source_documents"]

def format_sources(source_documents):
    lines, seen = [], set()
    for doc in source_documents:
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:120].replace("\n", " ").strip()
        key = (page, snippet[:40])
        if key not in seen:
            seen.add(key)
            lines.append(f"📄 Page {page}: *{snippet}…*")
    return "\n".join(lines)
