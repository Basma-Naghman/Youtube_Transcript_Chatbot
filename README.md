# 🎬 YouTube Chatbot using RAG

A conversational AI chatbot that lets you **chat with any YouTube video** using Retrieval-Augmented Generation (RAG). Simply provide a YouTube video ID and ask questions — the bot answers using only the video's transcript content.

---

## 🧠 How It Works

This project follows the classic **RAG pipeline** in 4 stages:

```
YouTube Transcript → Text Splitting → Embedding & Vector Store → Retrieval → LLM Generation
```

| Stage | Description |
|-------|-------------|
| **1. Indexing** | Fetches the YouTube transcript, splits it into chunks, embeds them using HuggingFace, and stores them in a FAISS vector store |
| **2. Retrieval** | On each user question, retrieves the top-k most relevant transcript chunks using similarity search |
| **3. Augmentation** | Injects retrieved chunks into a prompt template as context |
| **4. Generation** | Passes the augmented prompt to a Groq-hosted LLM for a grounded answer |

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│   FAISS Retriever   │  ◄── YouTube Transcript (chunked + embedded)
└─────────────────────┘
     │  Top-K Chunks
     ▼
┌─────────────────────┐
│   Prompt Template   │  ◄── "Answer only from the transcript..."
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│    Groq LLM (GPT)   │
└─────────────────────┘
     │
     ▼
   Answer
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `youtube-transcript-api` | Fetch YouTube video transcripts |
| `LangChain` | RAG pipeline orchestration |
| `FAISS` | Fast vector similarity search |
| `HuggingFace Inference API` | Text embeddings (`BAAI/bge-small-en-v1.5`) |
| `Groq` | Fast LLM inference |
| `Python` | Core language |

---

## 📦 Installation

```bash
pip install youtube-transcript-api langchain-community langchain-groq \
            langchain-core langchain-text-splitters faiss-cpu \
            huggingface_hub python-dotenv
```

---

## 🔑 Environment Variables

Create a `.env` file or set these in your environment:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

- Get your **Groq API key** → https://console.groq.com
- Get your **HuggingFace token** → https://huggingface.co/settings/tokens

---

## 🚀 Usage

```python
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# Step 1: Set your target YouTube video
video_id = "VITy7mDXHsQ"   # Replace with any YouTube video ID

# Step 2: Fetch transcript
api = YouTubeTranscriptApi()
transcript_list_obj = api.list(video_id)
transcript = transcript_list_obj.find_transcript(language_codes=['en'])
data = transcript.fetch()
text = " ".join(chunk.text for chunk in data)

# Step 3: Run the chain
main_chain.invoke("What is discussed in this video?")
```

---

## 💬 Example Queries

```python
main_chain.invoke("Who is the main speaker in this video?")
main_chain.invoke("What are the key points discussed?")
main_chain.invoke("Is nuclear function discussed in this video?")
```

---

## 📁 Project Structure

```
youtube-rag-chatbot/
│
├── youtude_chatbot_using_rag.py   # Main notebook/script
├── README.md                      # This file
├── .env                           # API keys (not committed to git)
└── requirements.txt               # Python dependencies
```

---

## ⚙️ Pipeline Details

### 1. Text Splitting
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.create_documents([text])
```

### 2. Embeddings
Uses `BAAI/bge-small-en-v1.5` via HuggingFace Inference API — a lightweight but powerful embedding model.

### 3. Vector Store
FAISS (Facebook AI Similarity Search) stores and retrieves embeddings locally in memory.

### 4. LLM
Groq's fast inference API powers the generation step. Currently configured with `openai/gpt-oss-120b`.

### 5. LangChain Chain
```python
main_chain = parallel_chain | prompt | model | parser
```

---

## ⚠️ Limitations

- Only works on YouTube videos that have **English captions enabled**
- Answers are strictly grounded in the transcript — the model will say *"I don't know"* if the answer isn't in the video
- Very long videos may produce many chunks and increase embedding time

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**[Basma Nghman]**
- GitHub: (https://github.com/Basma-Naghman)
- LinkedIn: ](https://www.linkedin.com/in/basma-naghman-50a925279/?locale=%20download)

---

> 💡 **Tip:** This project was built and tested on Google Colab. You can open it directly in Colab using the notebook link.
