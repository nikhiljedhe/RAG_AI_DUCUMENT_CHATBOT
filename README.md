# 📄 AI Document Q&A Chatbot


Chat with any PDF using AI — 100% free, no OpenAI key needed.Upload any PDF or text file, ask questions in plain English, and get accurate answers pulled directly from the document.No hallucinations, no guessing — the AI only answers from what's in your file.

# 🆓 Free Tech Stack

## LLM : Groq API (LLaMA 3)

## Embeddings : HuggingFace all-MiniLM-L6-v2

## Vector Search : FAISS

## UI : Streamlit

# 🏗️ How It Works
Upload a PDF or TXT file
The document gets split into chunks and each chunk gets converted into a numerical fingerprint (embedding)
When you ask a question, the app finds the most relevant chunks using FAISS similarity search
Those chunks are handed to LLaMA 3 via Groq which generates a grounded answer
You get the answer + source page references so you can verify it yourself

# 🚀 Run on Google Colab
You need two free API keys — no credit card required for either:

Groq API key → https://console.groq.com

ngrok token → https://dashboard.ngrok.com

Open RAG_Chatbot_Colab.ipynb and run all 4 cells in order. Your live app link prints at the end of Cell 4.

# 💻 Run Locally
Clone the repo and enter the folder
Create and activate a virtual environment
Run pip install -r requirements.txt
Copy .env.example to .env and paste your Groq key inside
Run streamlit run app.py
Open http://localhost:8501 in your browser

# ⚙️ Available Models
### All free via Groq:

llama-3.1-8b-instant — fastest

llama-3.3-70b-versatile — best quality

llama-4-scout-17b-16e-instruct — balanced

qwen/qwen3-32b — long documents

Switch between models from the sidebar inside the app.

# 🔧 Tips

Dense PDFs → set Chunk Size to 500

Narrative text → set Chunk Size to 1500

Incomplete answers → increase Top-K to 5 or 6

# ☁️ Deploy to Streamlit Cloud

Push this repo to GitHub

Go to https://share.streamlit.io and create a New App

Point it at your repo

Add GROQ_API_KEY under Settings → Secrets
