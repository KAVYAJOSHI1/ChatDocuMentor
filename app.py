import streamlit as st
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch
from urllib.parse import urlparse, urljoin
import time
import base64

# Set page config
st.set_page_config(
    page_title="🤖 AI Document Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-right: auto;
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00b894;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #e17055;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #74b9ff;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #667eea;
        padding: 0.5rem 1rem;
    }
    
    .stSelectbox > div > div > select {
        border-radius: 25px;
        border: 2px solid #667eea;
    }
    
    .stFileUploader > div > div > div {
        border-radius: 15px;
        border: 2px dashed #667eea;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []
if 'index' not in st.session_state:
    st.session_state.index = None

@st.cache_resource
def load_models():
    """Load and cache the ML models"""
    try:
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load QA model
        qa_model_name = "deepset/roberta-base-squad2"
        qa_pipeline = pipeline(
            "question-answering",
            model=qa_model_name,
            tokenizer=qa_model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
        return embedding_model, qa_pipeline
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def extract_text_from_url(url):
    """Extract text from a webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from URL: {e}")
        return None

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks for better processing"""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def create_embeddings(text_chunks, embedding_model):
    """Create embeddings for text chunks"""
    try:
        embeddings = embedding_model.encode(text_chunks)
        return embeddings
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None

def create_faiss_index(embeddings):
    """Create FAISS index for similarity search"""
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        return None

def find_relevant_chunks(query, embedding_model, index, text_chunks, k=3):
    """Find most relevant text chunks for a query"""
    try:
        query_embedding = embedding_model.encode([query])
        distances, indices = index.search(query_embedding.astype('float32'), k)
        
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(text_chunks):
                relevant_chunks.append({
                    'text': text_chunks[idx],
                    'score': distances[0][i]
                })
        
        return relevant_chunks
    except Exception as e:
        st.error(f"Error finding relevant chunks: {e}")
        return []

def answer_question(question, context, qa_pipeline):
    """Generate answer using QA model"""
    try:
        # Limit context length to avoid token limits
        max_context_length = 2000
        if len(context) > max_context_length:
            context = context[:max_context_length]
        
        result = qa_pipeline(question=question, context=context)
        
        # Add confidence threshold
        if result['score'] > 0.01:  # Minimum confidence threshold
            return result['answer'], result['score']
        else:
            return "I couldn't find a confident answer to your question in the provided document.", 0.0
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while processing your question.", 0.0

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Document Chatbot</h1>
    <p>Upload a PDF or provide a website URL to start chatting with your documents!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Document Input")
    
    # Model loading status
    with st.spinner("Loading AI models..."):
        embedding_model, qa_pipeline = load_models()
    
    if embedding_model and qa_pipeline:
        st.success("✅ Models loaded successfully!")
    else:
        st.error("❌ Failed to load models")
        st.stop()
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📄 Upload PDF", "🌐 Website URL"]
    )
    
    # Document processing
    if input_method == "📄 Upload PDF":
        uploaded_file = st.file_uploader(
            "Upload PDF file",
            type=['pdf'],
            help="Upload a PDF document to chat with"
        )
        
        if uploaded_file is not None:
            if st.button("📊 Process PDF"):
                with st.spinner("Processing PDF..."):
                    text = extract_text_from_pdf(uploaded_file)
                    if text:
                        st.session_state.text_chunks = chunk_text(text)
                        st.session_state.embeddings = create_embeddings(
                            st.session_state.text_chunks, embedding_model
                        )
                        if st.session_state.embeddings is not None:
                            st.session_state.index = create_faiss_index(st.session_state.embeddings)
                            st.session_state.document_processed = True
                            st.success("✅ PDF processed successfully!")
                        else:
                            st.error("❌ Failed to process PDF")
    
    else:  # Website URL
        url = st.text_input(
            "Enter website URL:",
            placeholder="https://example.com"
        )
        
        if st.button("🔍 Process Website"):
            if url:
                with st.spinner("Extracting content from website..."):
                    text = extract_text_from_url(url)
                    if text:
                        st.session_state.text_chunks = chunk_text(text)
                        st.session_state.embeddings = create_embeddings(
                            st.session_state.text_chunks, embedding_model
                        )
                        if st.session_state.embeddings is not None:
                            st.session_state.index = create_faiss_index(st.session_state.embeddings)
                            st.session_state.document_processed = True
                            st.success("✅ Website processed successfully!")
                        else:
                            st.error("❌ Failed to process website")
            else:
                st.warning("Please enter a valid URL")
    
    # Document info
    if st.session_state.document_processed:
        st.markdown("### 📈 Document Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(st.session_state.text_chunks)}</h3>
                <p>Text Chunks</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(' '.join(st.session_state.text_chunks).split())}</h3>
                <p>Total Words</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear document button
        if st.button("🗑️ Clear Document"):
            st.session_state.document_processed = False
            st.session_state.messages = []
            st.session_state.text_chunks = []
            st.session_state.embeddings = None
            st.session_state.index = None
            st.rerun()

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    if st.session_state.document_processed:
        st.markdown("### 💬 Chat with your document")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bot-message">
                        <strong>AI:</strong> {message["content"]}
                        {f"<br><small>Confidence: {message.get('confidence', 0):.2%}</small>" if message.get('confidence') else ""}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            col_input, col_button = st.columns([4, 1])
            with col_input:
                user_question = st.text_input(
                    "Ask a question about your document:",
                    placeholder="What is this document about?",
                    label_visibility="collapsed"
                )
            with col_button:
                submit_button = st.form_submit_button("Send 📤")
        
        if submit_button and user_question:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # Find relevant chunks
            relevant_chunks = find_relevant_chunks(
                user_question, 
                embedding_model, 
                st.session_state.index, 
                st.session_state.text_chunks
            )
            
            # Create context from relevant chunks
            context = " ".join([chunk['text'] for chunk in relevant_chunks])
            
            # Generate answer
            answer, confidence = answer_question(user_question, context, qa_pipeline)
            
            # Add AI response
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "confidence": confidence
            })
            
            # Rerun to update chat
            st.rerun()
    
    else:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Getting Started</h3>
            <p>To begin chatting with your documents:</p>
            <ol>
                <li>📄 Upload a PDF file or 🌐 enter a website URL in the sidebar</li>
                <li>📊 Click the process button to analyze your document</li>
                <li>💬 Start asking questions about your document!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Example questions
        st.markdown("### 🤔 Example Questions You Can Ask")
        example_questions = [
            "What is the main topic of this document?",
            "Can you summarize the key points?",
            "What are the important dates mentioned?",
            "Who are the main people or organizations discussed?",
            "What conclusions does the document reach?"
        ]
        
        for question in example_questions:
            st.markdown(f"• {question}")

with col2:
    st.markdown("### 🎯 Features")
    features = [
        "🔍 Smart document processing",
        "🧠 Advanced AI understanding",
        "⚡ Fast similarity search",
        "📊 Confidence scoring",
        "🌐 Website content extraction",
        "📄 PDF text extraction",
        "💬 Interactive chat interface"
    ]
    
    for feature in features:
        st.markdown(f"• {feature}")
    
    # Model info
    st.markdown("### 🤖 AI Models Used")
    st.markdown("""
    **Embedding Model:** 
    - all-MiniLM-L6-v2
    - Fast and efficient sentence embeddings
    
    **QA Model:**
    - RoBERTa-base-squad2
    - State-of-the-art question answering
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with ❤️ using Streamlit, Hugging Face Transformers, and FAISS</p>
    <p>Powered by advanced AI models for document understanding</p>
</div>
""", unsafe_allow_html=True)
