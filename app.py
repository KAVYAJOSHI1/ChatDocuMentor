import streamlit as st
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
from urllib.parse import urlparse
import time
import logging
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

# Enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced NLTK handling
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    def sent_tokenize(text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def word_tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

# Enhanced page configuration
st.set_page_config(
    page_title="🚀 AI Document Chatbot Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# AI Document Chatbot Pro\nPowered by advanced AI models for maximum accuracy!"
    }
)

# Modern CSS with animations and glassmorphism
modern_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-size: 400% 400%;
        animation: gradientShift 8s ease infinite;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shine 3s infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes shine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    .chat-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        max-height: 500px;
        overflow-y: auto;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .message-container {
        margin: 1rem 0;
        animation: fadeInUp 0.5s ease-out;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 25px 25px 8px 25px;
        margin: 1rem 0;
        max-width: 85%;
        margin-left: auto;
        word-wrap: break-word;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        position: relative;
        transform: translateX(0);
        transition: all 0.3s ease;
    }

    .user-message:hover {
        transform: translateX(-5px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
    }

    .bot-message {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 25px 25px 25px 8px;
        margin: 1rem 0;
        max-width: 85%;
        margin-right: auto;
        word-wrap: break-word;
        box-shadow: 0 8px 25px rgba(116, 185, 255, 0.3);
        position: relative;
        transform: translateX(0);
        transition: all 0.3s ease;
    }

    .bot-message:hover {
        transform: translateX(5px);
        box-shadow: 0 12px 35px rgba(116, 185, 255, 0.4);
    }

    .confidence-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
        backdrop-filter: blur(5px);
    }

    .confidence-high {
        background: rgba(0, 184, 148, 0.8);
        color: white;
    }

    .confidence-medium {
        background: rgba(253, 203, 110, 0.8);
        color: white;
    }

    .confidence-low {
        background: rgba(225, 112, 85, 0.8);
        color: white;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px) scale(1.02);
    }

    .success-alert {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #2d3436;
        padding: 1rem;
        border-radius: 15px;
        border-left: 5px solid #00b894;
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }

    .error-alert {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #2d3436;
        padding: 1rem;
        border-radius: 15px;
        border-left: 5px solid #e17055;
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }

    .info-alert {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2d3436;
        padding: 1rem;
        border-radius: 15px;
        border-left: 5px solid #74b9ff;
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 0.8rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }

    .feature-list {
        list-style: none;
        padding: 0;
    }

    .feature-item {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }

    .feature-item:hover {
        transform: translateX(5px);
        background: rgba(255, 255, 255, 0.15);
    }

    .loading-spinner {
        border: 4px solid rgba(102, 126, 234, 0.3);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }

    .status-online {
        background: #00b894;
    }

    .status-offline {
        background: #e17055;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
"""

st.markdown(modern_css, unsafe_allow_html=True)

# Enhanced session state initialization
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'messages': [],
        'document_processed': False,
        'embeddings': None,
        'text_chunks': [],
        'index': None,
        'raw_text': "",
        'processing_time': 0,
        'model_status': 'loading',
        'document_stats': {}
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# Enhanced model loading with better error handling
@st.cache_resource(show_spinner=False)
def load_advanced_models():
    """Load state-of-the-art models with comprehensive error handling"""
    models = {}
    load_status = {}

    try:
        # 1. Load the best embedding model
        with st.spinner("🧠 Loading semantic embedding model..."):
            try:
                models['embedding'] = SentenceTransformer('all-MiniLM-L6-v2')
                load_status['embedding'] = 'success'
                logger.info("Successfully loaded embedding model")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                load_status['embedding'] = 'failed'
                models['embedding'] = None

        # 2. Load advanced QA pipeline
        with st.spinner("🎯 Loading question-answering model..."):
            qa_models = [
                "deepset/roberta-base-squad2",
                "distilbert-base-cased-distilled-squad",
                "bert-large-uncased-whole-word-masking-finetuned-squad"
            ]

            for model_name in qa_models:
                try:
                    models['qa'] = pipeline(
                        "question-answering",
                        model=model_name,
                        tokenizer=model_name,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    load_status['qa'] = f'success - {model_name}'
                    logger.info(f"Successfully loaded QA model: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load QA model {model_name}: {e}")
                    continue

            if 'qa' not in models:
                load_status['qa'] = 'failed'
                models['qa'] = None

        # 3. Load summarization model for better context understanding
        with st.spinner("📝 Loading summarization model..."):
            try:
                models['summarizer'] = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=200,
                    min_length=50,
                    do_sample=False
                )
                load_status['summarizer'] = 'success'
                logger.info("Successfully loaded summarization model")
            except Exception as e:
                logger.warning(f"Failed to load summarization model: {e}")
                load_status['summarizer'] = 'failed'
                models['summarizer'] = None

        return models, load_status

    except Exception as e:
        logger.error(f"Critical error in model loading: {e}")
        return {}, {'error': str(e)}

# Enhanced text preprocessing
def advanced_text_preprocessing(text: str) -> str:
    """Advanced text preprocessing with multiple techniques"""
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Fix common PDF extraction issues
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces

    # Clean up special characters while preserving important punctuation
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\'\"\(\)\[\]\/\@\#\$\%\&\*\+\=]', ' ', text)

    # Normalize sentence boundaries
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)

    # Remove very short sentences that are likely noise
    sentences = sent_tokenize(text)
    cleaned_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 15 and len(sentence.split()) > 3:
            cleaned_sentences.append(sentence)

    return ' '.join(cleaned_sentences)

# Enhanced web scraping
def extract_web_content(url: str) -> Optional[str]:
    """Enhanced web content extraction with better error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside',
                             'menu', 'form', 'button', 'noscript', 'iframe', 'embed']):
            element.decompose()

        # Try to find main content using multiple strategies
        content_selectors = [
            'article', 'main', '[role="main"]', '.content', '.main-content',
            '.post-content', '.entry-content', '.article-content', '.page-content',
            '#content', '#main', '#primary'
        ]

        extracted_text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                extracted_text = ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
                break

        if not extracted_text:
            extracted_text = soup.get_text(separator=' ', strip=True)

        processed_text = advanced_text_preprocessing(extracted_text)

        if len(processed_text.split()) < 50:
            st.warning("⚠️ Limited content extracted. The website might have restrictions or minimal text content.")

        return processed_text

    except requests.exceptions.Timeout:
        st.error("❌ Request timeout. The website took too long to respond.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection error. Please check your internet connection.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error extracting web content: {e}")
        return None

# Enhanced PDF processing
def extract_pdf_content(pdf_file) -> Optional[str]:
    """Enhanced PDF content extraction with better error handling"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)

        if total_pages == 0:
            st.error("❌ PDF file appears to be empty.")
            return None

        extracted_text = ""
        successful_pages = 0

        # Progress bar for PDF processing
        progress_bar = st.progress(0)

        for page_num in range(total_pages):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()

                if page_text and len(page_text.strip()) > 10:
                    extracted_text += page_text + "\n"
                    successful_pages += 1

                progress_bar.progress((page_num + 1) / total_pages)

            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue

        progress_bar.empty()

        if successful_pages == 0:
            st.error("❌ Could not extract text from any pages. The PDF might be image-based or corrupted.")
            return None

        if successful_pages < total_pages:
            st.warning(f"⚠️ Successfully extracted text from {successful_pages} out of {total_pages} pages.")

        processed_text = advanced_text_preprocessing(extracted_text)

        if len(processed_text.split()) < 50:
            st.warning("⚠️ Limited text extracted from PDF. Consider using OCR for image-based PDFs.")

        return processed_text

    except PyPDF2.errors.PdfReadError:
        st.error("❌ PDF file is corrupted or encrypted. Please try a different file.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error processing PDF: {e}")
        return None

# Intelligent chunking with overlap
def intelligent_text_chunking(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Advanced text chunking with semantic awareness"""
    if not text:
        return []

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if current_size + sentence_words > max_chunk_size and current_chunk:
            # Create chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

            # Create overlap by keeping last few sentences
            overlap_sentences = []
            overlap_words = 0

            for i in range(len(current_chunk) - 1, -1, -1):
                sent = current_chunk[i]
                sent_words = len(sent.split())
                if overlap_words + sent_words <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_words += sent_words
                else:
                    break

            current_chunk = overlap_sentences + [sentence]
            current_size = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_size += sentence_words

    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    # Filter out very short chunks
    quality_chunks = [chunk for chunk in chunks if len(chunk.split()) > 30]

    return quality_chunks

# Enhanced embedding creation
def create_semantic_embeddings(text_chunks: List[str], embedding_model) -> Optional[np.ndarray]:
    """Create high-quality semantic embeddings"""
    if not text_chunks or not embedding_model:
        return None

    try:
        batch_size = 32
        all_embeddings = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]

            status_text.text(f"Creating embeddings... {i + len(batch)}/{len(text_chunks)}")

            batch_embeddings = embedding_model.encode(
                batch,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True
            )

            all_embeddings.extend(batch_embeddings)
            progress_bar.progress(min((i + batch_size) / len(text_chunks), 1.0))

        progress_bar.empty()
        status_text.empty()

        return np.array(all_embeddings)

    except Exception as e:
        st.error(f"❌ Error creating embeddings: {e}")
        return None

# Enhanced FAISS index creation
def create_vector_index(embeddings: np.ndarray) -> Optional[faiss.Index]:
    """Create optimized FAISS index for similarity search"""
    if embeddings is None or len(embeddings) == 0:
        return None

    try:
        dimension = embeddings.shape[1]

        # Use IndexFlatIP for cosine similarity
        index = faiss.IndexFlatIP(dimension)

        # Add embeddings to index
        index.add(embeddings.astype('float32'))

        return index

    except Exception as e:
        st.error(f"❌ Error creating search index: {e}")
        return None

# Enhanced context retrieval
def retrieve_relevant_context(query: str, embedding_model, index, text_chunks: List[str], k: int = 5) -> List[Dict]:
    """Retrieve most relevant context with enhanced scoring"""
    if not query or not index or not text_chunks:
        return []

    try:
        # Create query embedding
        query_embedding = embedding_model.encode([query], normalize_embeddings=True)

        # Search for similar chunks
        actual_k = min(k, len(text_chunks))
        scores, indices = index.search(query_embedding.astype('float32'), actual_k)

        relevant_context = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(text_chunks) and score > 0.2:  # Threshold for relevance
                relevant_context.append({
                    'text': text_chunks[idx],
                    'score': float(score),
                    'index': int(idx)
                })

        # Sort by relevance score
        relevant_context.sort(key=lambda x: x['score'], reverse=True)

        return relevant_context

    except Exception as e:
        st.error(f"❌ Error retrieving context: {e}")
        return []

# Enhanced answer generation
def generate_intelligent_answer(question: str, context_chunks: List[Dict], qa_pipeline, summarizer=None) -> Tuple[str, float]:
    """Generate intelligent answers using multiple AI techniques"""
    if not context_chunks:
        return "I couldn't find relevant information in the document to answer your question.", 0.0

    try:
        # Combine top context chunks
        combined_context = " ".join([chunk['text'] for chunk in context_chunks[:3]])

        # Truncate context if too long
        max_context_length = 2048
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length] + "..."

        # Primary QA approach
        qa_result = qa_pipeline(
            question=question,
            context=combined_context,
            max_answer_len=200,
            handle_impossible_answer=True
        )

        answer = qa_result['answer']
        confidence = qa_result['score']

        # Enhance answer if confidence is low
        if confidence < 0.5 and summarizer:
            try:
                # Use summarization to provide broader context
                summary_input = f"Question: {question}\n\nContext: {combined_context}"
                if len(summary_input) > 1024:
                    summary_input = summary_input[:1024] + "..."

                summary = summarizer(summary_input, max_length=150, min_length=50, do_sample=False)
                enhanced_answer = summary[0]['summary_text']

                if len(enhanced_answer) > 20:
                    answer = enhanced_answer
                    confidence = 0.6  # Moderate confidence for summarized answers

            except Exception as e:
                logger.warning(f"Summarization failed: {e}")

        # Fallback for very low confidence
        if confidence < 0.3:
            # Extract most relevant sentences
            relevant_sentences = []
            for chunk in context_chunks[:2]:
                sentences = sent_tokenize(chunk['text'])
                for sentence in sentences:
                    if any(word.lower() in sentence.lower() for word in question.split() if len(word) > 3):
                        relevant_sentences.append(sentence)
                        if len(relevant_sentences) >= 3:
                            break
                if len(relevant_sentences) >= 3:
                    break

            if relevant_sentences:
                answer = "Based on the document: " + " ".join(relevant_sentences)
                confidence = 0.4

        return answer, confidence

    except Exception as e:
        st.error(f"❌ Error generating answer: {e}")
        return "I encountered an error while processing your question.", 0.0

# Utility functions
def get_confidence_info(confidence: float) -> Tuple[str, str]:
    """Get confidence level and color class"""
    if confidence >= 0.7:
        return "High", "confidence-high"
    elif confidence >= 0.5:
        return "Medium", "confidence-medium"
    else:
        return "Low", "confidence-low"

def calculate_document_stats(text: str) -> Dict:
    """Calculate comprehensive document statistics"""
    if not text:
        return {}

    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    stats = {
        'total_words': len(words),
        'total_sentences': len(sentences),
        'total_characters': len(text),
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
        'reading_time': len(words) / 200,  # Assuming 200 words per minute
        'vocabulary_size': len(set(words))
    }

    return stats

def display_chat_message(message: Dict, is_user: bool = False):
    """Display a chat message with proper styling"""
    if is_user:
        st.markdown(f"""
        <div class="message-container">
            <div class="user-message">
                <strong>You:</strong> {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        confidence_level, confidence_class = get_confidence_info(message.get('confidence', 0.0))
        st.markdown(f"""
        <div class="message-container">
            <div class="bot-message">
                <strong>AI Assistant:</strong> {message['content']}
                <div class="confidence-badge {confidence_class}">
                    Confidence: {confidence_level} ({message.get('confidence', 0.0):.2f})
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Moved these function definitions before main() to ensure they are defined before being called
def process_document(content: str, models: Dict):
    """Process document and create embeddings"""
    if not content or not models.get('embedding'):
        st.error("❌ Cannot process document. Please check your input and model status.")
        return

    start_time = time.time()

    try:
        # Store raw text
        st.session_state.raw_text = content

        # Calculate document statistics
        st.session_state.document_stats = calculate_document_stats(content)

        # Create text chunks
        with st.spinner("🔧 Creating text chunks..."):
            text_chunks = intelligent_text_chunking(content)

            if not text_chunks:
                st.error("❌ Failed to create text chunks. Please check your document content.")
                return

            st.session_state.text_chunks = text_chunks

        # Create embeddings
        with st.spinner("🧠 Creating semantic embeddings..."):
            embeddings = create_semantic_embeddings(text_chunks, models['embedding'])

            if embeddings is None:
                st.error("❌ Failed to create embeddings.")
                return

            st.session_state.embeddings = embeddings

        # Create search index
        with st.spinner("🔍 Building search index..."):
            index = create_vector_index(embeddings)

            if index is None:
                st.error("❌ Failed to create search index.")
                return

            st.session_state.index = index

        # Mark as processed
        st.session_state.document_processed = True
        st.session_state.processing_time = time.time() - start_time

        # Show success message
        st.markdown(f"""
        <div class="success-alert">
            <h4>✅ Document processed successfully!</h4>
            <p>
                • Processed {len(text_chunks)} text chunks<br>
                • Created {len(embeddings)} semantic embeddings<br>
                • Processing time: {st.session_state.processing_time:.2f} seconds
            </p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error processing document: {e}")
        logger.error(f"Document processing error: {e}")

def generate_response(question: str, models: Dict) -> Dict:
    """Generate intelligent response to user question"""
    if not st.session_state.document_processed:
        return {
            'answer': "Please upload and process a document first.",
            'confidence': 0.0
        }

    try:
        # Retrieve relevant context
        context_chunks = retrieve_relevant_context(
            question,
            models['embedding'],
            st.session_state.index,
            st.session_state.text_chunks,
            k=5
        )

        if not context_chunks:
            return {
                'answer': "I couldn't find relevant information in the document to answer your question. Please try rephrasing your question or ask about different aspects of the document.",
                'confidence': 0.0
            }

        # Generate answer
        answer, confidence = generate_intelligent_answer(
            question,
            context_chunks,
            models['qa'],
            models.get('summarizer')
        )

        return {
            'answer': answer,
            'confidence': confidence
        }

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            'answer': "I encountered an error while processing your question. Please try again.",
            'confidence': 0.0
        }


# Main application
def main():
    """Main application function"""

    # Modern header with animations
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI Document Chatbot Pro</h1>
        <p style="font-size: 1.2rem; opacity: 0.9; margin-top: 1rem;">
            Powered by advanced AI models for intelligent document analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    with st.spinner("🤖 Initializing AI models..."):
        models, load_status = load_advanced_models()

        if models.get('embedding') and models.get('qa'):
            st.session_state.model_status = 'ready'
            st.success("✅ All AI models loaded successfully!")
        else:
            st.session_state.model_status = 'error'
            st.error("❌ Failed to load some AI models. Please check your configuration.")

    # Sidebar for document upload and configuration
    with st.sidebar:
        st.markdown("## 📁 Document Upload")

        # Document input options
        input_method = st.radio(
            "Choose input method:",
            ["📄 Upload PDF", "🌐 Web URL", "✍️ Direct Text"],
            help="Select how you want to provide the document"
        )

        document_content = None

        if input_method == "📄 Upload PDF":
            uploaded_file = st.file_uploader(
                "Upload a PDF document",
                type=['pdf'],
                help="Upload a PDF file to analyze"
            )

            if uploaded_file is not None:
                with st.spinner("🔍 Extracting PDF content..."):
                    document_content = extract_pdf_content(uploaded_file)

        elif input_method == "🌐 Web URL":
            url = st.text_input(
                "Enter website URL:",
                placeholder="https://example.com/article",
                help="Enter a valid URL to extract content"
            )

            if url and st.button("🔍 Extract Content"):
                if urlparse(url).scheme in ['http', 'https']:
                    with st.spinner("🌐 Extracting web content..."):
                        document_content = extract_web_content(url)
                else:
                    st.error("❌ Please enter a valid URL starting with http:// or https://")

        elif input_method == "✍️ Direct Text":
            document_content = st.text_area(
                "Paste your text here:",
                height=200,
                placeholder="Enter or paste your document content here...",
                help="Paste the text you want to analyze"
            )

        # Process document
        if document_content and st.button("🚀 Process Document"):
            process_document(document_content, models)

        # Display model status
        st.markdown("## 🤖 AI Model Status")

        if st.session_state.model_status == 'ready':
            st.markdown('<div class="status-indicator status-online"></div>Models Ready', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-offline"></div>Models Loading', unsafe_allow_html=True)

        # Display document stats
        if st.session_state.document_stats:
            st.markdown("## 📊 Document Statistics")
            stats = st.session_state.document_stats

            col1, col2 = st.columns(2)
            with col1:
                st.metric("📝 Words", f"{stats.get('total_words', 0):,}")
                st.metric("📖 Sentences", f"{stats.get('total_sentences', 0):,}")

            with col2:
                st.metric("⏱️ Reading Time", f"{stats.get('reading_time', 0):.1f} min")
                st.metric("📚 Vocabulary", f"{stats.get('vocabulary_size', 0):,}")

    # Main chat interface
    if st.session_state.document_processed:
        st.markdown("## 💬 Chat with your Document")

        # Display chat history
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)

            for message in st.session_state.messages:
                display_chat_message(message, message['role'] == 'user')

            st.markdown('</div>', unsafe_allow_html=True)

        # Chat input
        user_question = st.text_input(
            "Ask a question about your document:",
            placeholder="What is the main topic of this document?",
            key="user_input"
        )

        if user_question:
            # Add user message to chat
            st.session_state.messages.append({
                'role': 'user',
                'content': user_question,
                'timestamp': datetime.now().isoformat()
            })

            # Generate response
            with st.spinner("🤔 Thinking..."):
                response = generate_response(user_question, models)

                # Add bot response to chat
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': response['answer'],
                    'confidence': response['confidence'],
                    'timestamp': datetime.now().isoformat()
                })

            # Refresh the page to show new messages
            st.rerun()

        # Quick actions
        st.markdown("### 🎯 Quick Actions")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📋 Summarize Document"):
                with st.spinner("📝 Creating summary..."):
                    summary_response = generate_response("Please provide a comprehensive summary of this document.", models)
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': summary_response['answer'],
                        'confidence': summary_response['confidence'],
                        'timestamp': datetime.now().isoformat()
                    })
                    st.rerun()

        with col2:
            if st.button("🔑 Key Points"):
                with st.spinner("🔍 Extracting key points..."):
                    key_points_response = generate_response("What are the main key points and important information in this document?", models)
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': key_points_response['answer'],
                        'confidence': key_points_response['confidence'],
                        'timestamp': datetime.now().isoformat()
                    })
                    st.rerun()

        with col3:
            if st.button("🏷️ Topics"):
                with st.spinner("🎯 Identifying topics..."):
                    topics_response = generate_response("What are the main topics and themes discussed in this document?", models)
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': topics_response['answer'],
                        'confidence': topics_response['confidence'],
                        'timestamp': datetime.now().isoformat()
                    })
                    st.rerun()

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    else:
        # Welcome screen
        st.markdown("## 🌟 Welcome to AI Document Chatbot Pro!")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="glass-card">
                <h3>🚀 Features</h3>
                <ul class="feature-list">
                    <li class="feature-item">📄 PDF Document Processing</li>
                    <li class="feature-item">🌐 Web Content Extraction</li>
                    <li class="feature-item">🧠 Advanced AI Understanding</li>
                    <li class="feature-item">💬 Interactive Chat Interface</li>
                    <li class="feature-item">📊 Document Analytics</li>
                    <li class="feature-item">🎯 Smart Question Answering</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="glass-card">
                <h3>🔧 How to Use</h3>
                <ol>
                    <li>Choose your input method in the sidebar</li>
                    <li>Upload a PDF, enter a URL, or paste text</li>
                    <li>Click "Process Document" to analyze</li>
                    <li>Start asking questions about your document</li>
                    <li>Use quick actions for common tasks</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

        # Sample questions for inspiration
        st.markdown("## 💡 Example Questions You Can Ask")

        example_questions = [
            "What is the main topic of this document?",
            "Can you summarize the key points?",
            "What are the most important findings?",
            "Are there any specific dates or numbers mentioned?",
            "Who are the main people or organizations discussed?",
            "What conclusions does the document reach?"
        ]

        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                st.markdown(f"**{i+1}.** {question}")

# Run the application
if __name__ == "__main__":
    main()
