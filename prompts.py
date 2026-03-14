
from langchain.prompts import PromptTemplate

TEMPLATE = """You are a precise document assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say: "I could not find this information in the document."

Context:
{context}

Question: {question}

Answer (be concise and specific):"""

QA_PROMPT = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])
