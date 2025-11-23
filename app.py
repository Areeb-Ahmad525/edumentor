import streamlit as st
import os, re, glob, shutil
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from groq import Groq
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================
# 1. CONFIG & UI STYLING
# ==========================================
st.set_page_config(page_title="EduMentor AI", page_icon="🎓", layout="wide")

# Custom CSS for a "Decent & Beautiful" look
st.markdown("""
    <style>
    /* Main Background and Text */
    .stApp {
        background-color: #f9f9f9;
        color: #333333;
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e6e6e6;
    }
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        color: #1a1a1a;
    }
    /* Chat Bubbles */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 15px;
        border: 1px solid #eee;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    /* Button Styling */
    div.stButton > button {
        background-color: #2e86de;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #0984e3;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BACKEND LOGIC (Adapted from your script)
# ==========================================

PERSIST_DIR = "./edumentor_memory"

def extract_video_id(url):
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]+)", url)
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
        resp = requests.get(url, timeout=12)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            captions = soup.find_all("div", {"class": "caption-line"})
            if captions:
                return "\n".join([c.get_text(strip=True) for c in captions])
    except:
        pass
    return None

def download_audio(url, cookies_path=None):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "audio.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}],
    }
    if cookies_path:
        ydl_opts["cookiefile"] = cookies_path

    # Cleanup old files
    if os.path.exists("audio.mp3"): os.remove("audio.mp3")

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "audio.mp3" if os.path.exists("audio.mp3") else None

def whisper_transcribe(client, audio_file):
    # Simple wrapper for the logic in your script
    def transcribe_file(path):
        with open(path, "rb") as f:
            return client.audio.transcriptions.create(file=(path, f.read()), model="whisper-large-v3").text

    try:
        return transcribe_file(audio_file)
    except Exception as e:
        if "413" in str(e): # Too large
            # Compression
            os.system(f"ffmpeg -i {audio_file} -ar 16000 -ac 1 -b:a 32k compressed.mp3 -y")
            try:
                return transcribe_file("compressed.mp3")
            except:
                # Chunking
                os.system(f"ffmpeg -i compressed.mp3 -f segment -segment_time 600 -c copy chunk_%03d.mp3 -y")
                full_text = ""
                for c in sorted(glob.glob("chunk_*.mp3")):
                    full_text += transcribe_file(c) + " "
                return full_text
        raise e

@st.cache_resource
def setup_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vector_store(text):
    # Reset DB for new video
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_text(text)
    embed = setup_embedding_model()
    vectordb = Chroma.from_texts(texts=chunks, embedding=embed, persist_directory=PERSIST_DIR)
    return vectordb

def get_mentor_response(client, question, vectordb):
    docs = vectordb.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""
    You are an educational AI mentor. Use the context below to answer the student's question clearly and encouragingly.

    CONTEXT:
    {context}

    QUESTION: {question}

    ANSWER:
    """
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
    )
    return resp.choices[0].message.content

# ==========================================
# 3. FRONTEND LAYOUT
# ==========================================

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.markdown("---")
    st.subheader("🍪 Optional Cookies")
    cookies_file = st.file_uploader("Upload cookies.txt (for age-restricted videos)", type=["txt"])
    cookies_path = None
    if cookies_file:
        with open("cookies.txt", "wb") as f:
            f.write(cookies_file.getbuffer())
        cookies_path = "cookies.txt"
        st.success("Cookies loaded!")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Area ---
st.title("🎓 EduMentor AI")
st.caption("Turn any YouTube video into an interactive learning session.")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "current_video" not in st.session_state:
    st.session_state.current_video = None

# Video Input Section
col1, col2 = st.columns([3, 1])
with col1:
    video_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
with col2:
    process_btn = st.button("Analyze Video", use_container_width=True)

# Processing Logic
if process_btn and video_url and groq_key:
    st.session_state.current_video = video_url
    client = Groq(api_key=groq_key)

    with st.status("🚀 Processing content...", expanded=True) as status:
        try:
            # 1. Get Transcript
            st.write("📥 Extracting transcript...")
            transcript = None
            vid_id = extract_video_id(video_url)

            # Method A: API
            transcript = get_transcript_api(vid_id)

            # Method B: Scraper
            if not transcript:
                st.write("⚠️ API failed, trying scraper...")
                transcript = get_transcript_scraper(vid_id)

            # Method C: Whisper
            if not transcript:
                st.write("🔊 Using Whisper AI (Audio processing)...")
                audio_file = download_audio(video_url, cookies_path)
                if audio_file:
                    transcript = whisper_transcribe(client, audio_file)

            if not transcript:
                status.update(label="❌ Failed to get transcript.", state="error")
                st.error("Could not extract text from this video.")
                st.stop()

            # 2. Build Memory
            st.write("🧠 Building Knowledge Base...")
            st.session_state.vectordb = build_vector_store(transcript)

            status.update(label="✅ Ready to teach!", state="complete")
            st.session_state.messages = [{"role": "assistant", "content": "I've watched the video! What would you like to know?"}]

        except Exception as e:
            status.update(label="❌ Error occurred", state="error")
            st.error(f"Error: {str(e)}")

elif process_btn and not groq_key:
    st.warning("Please enter your Groq API Key in the sidebar.")

# Chat Interface
if st.session_state.vectordb:
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle User Input
    if prompt := st.chat_input("Ask a question about the video..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                client = Groq(api_key=groq_key)
                response = get_mentor_response(client, prompt, st.session_state.vectordb)
                st.markdown(response)

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    if not process_btn:
        st.info("Paste a URL above and click 'Analyze Video' to start.")
