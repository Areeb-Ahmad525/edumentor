import streamlit as st
import os
import re
import shutil
import glob
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from groq import Groq
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================================================
# 1. PAGE SETTINGS
# =========================================================

st.set_page_config(page_title="EduMentor AI", page_icon="🎓", layout="wide")
st.title("🎓 EduMentor AI")
st.caption("Turn any YouTube video into an interactive learning session.")

PERSIST_DIR = "edumentor_memory"

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def extract_video_id(url):
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})", url)
    return m.group(1) if m else None


def get_transcript_api(video_id):
    try:
        parts = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([p["text"] for p in parts])
    except:
        return None


def get_transcript_scraper(video_id):
    try:
        url = f"https://ytscribe.com/v/{video_id}"
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        caps = soup.find_all("div", {"class": "caption-line"})
        if caps:
            return "\n".join([c.get_text(strip=True) for c in caps])
    except:
        return None

    return None


def download_audio(url):
    """Streamlit Cloud cannot run Whisper; audio download is optional."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "audio.%(ext)s",
        "quiet": True,
        "noplaylist": True,
    }

    if os.path.exists("audio.mp3"):
        os.remove("audio.mp3")

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return "audio.mp3"
    except:
        return None


@st.cache_resource
def setup_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(text):
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embed = setup_embedding_model()
    vectordb = Chroma.from_texts(texts=chunks, embedding=embed, persist_directory=PERSIST_DIR)

    return vectordb


def get_ai_answer(client, question, vectordb):
    docs = vectordb.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an educational AI mentor. Use the following context to answer clearly.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600
    )
    return resp.choices[0].message.content


# =========================================================
# 3. SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Settings")

    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat cleared!")

# =========================================================
# 4. SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# =========================================================
# 5. VIDEO INPUT
# =========================================================

video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Analyze Video"):
    if not groq_key:
        st.warning("Please enter Groq API Key first.")
        st.stop()

    client = Groq(api_key=groq_key)

    with st.status("Processing Video...", expanded=True) as status:
        vid = extract_video_id(video_url)
        if not vid:
            status.update(label="❌ Invalid YouTube URL", state="error")
            st.stop()

        st.write("📥 Extracting transcript (API)...")
        transcript = get_transcript_api(vid)

        if not transcript:
            st.write("📥 Trying transcript scraper...")
            transcript = get_transcript_scraper(vid)

        if not transcript:
            status.update(label="❌ Transcript not found.", state="error")
            st.error("Video does NOT have a transcript. Cannot continue.")
            st.stop()

        st.write("🧠 Building vector database...")
        st.session_state.vectordb = build_vector_store(transcript)

        status.update(label="✅ Video processed successfully!", state="complete")

        st.session_state.messages = []
        st.success("Ask anything about the video now!")

# =========================================================
# 6. CHAT UI
# =========================================================

if st.session_state.vectordb:
    # show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask something about the video...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                client = Groq(api_key=groq_key)
                answer = get_ai_answer(client, user_input, st.session_state.vectordb)
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Paste a YouTube URL above and click 'Analyze Video' to begin.")
