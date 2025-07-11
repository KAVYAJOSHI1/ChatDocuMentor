import streamlit as st
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, T5ForConditionalGeneration, T5Tokenizer
import torch
from urllib.parse import urlparse, urljoin
import time
import base64
import logging

# Ensure NLTK downloads are handled
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    # Fallback sentence tokenizer
    def sent_tokenize(text):
        """Simple sentence tokenizer fallback"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text) # Improved regex for sentence splitting
        return [s.strip() for s in sentences if s.strip()]
    
    def word_tokenize(text):
        """Simple word tokenizer fallback"""
        import re
        return re.findall(r'\b\w+\b', text.lower())

# Set page config
st.set_page_config(
    page_title="🤖 AI Document Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
custom_css = """
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
        max-height: 400px;
        overflow-y: auto;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-right: auto;
        word-wrap: break-word;
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
        width: 100%;
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
    
    .confidence-high { color: #00b894; font-weight: bold; }
    .confidence-medium { color: #fdcb6e; font-weight: bold; }
    .confidence-low { color: #e17055; font-weight: bold; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

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
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = ""

@st.cache_resource
def load_models():
    """Load and cache the best ML models for accuracy"""
    try:
        # Use better embedding model for improved semantic understanding
        embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Use more powerful QA models
        qa_model_options = [
            "deepset/roberta-base-squad2",  # Robust QA model
            "distilbert-base-cased-distilled-squad"  # Faster fallback
        ]
        
        qa_pipeline = None
        for model_name in qa_model_options:
            try:
                qa_pipeline = pipeline(
                    "question-answering",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=512,
                    truncation=True
                )
                st.success(f"✅ Loaded QA model: **{model_name}**")
                break
            except Exception as e:
                st.warning(f"Failed to load {model_name}, trying next... Error: {e}")
                continue
        
        if qa_pipeline is None:
            raise Exception("Could not load any QA model")
        
        # Load a powerful generative model for better answers
        generative_pipeline = None
        try:
            # Using 'google/flan-t5-base' as it's a good general-purpose T5 model
            # and often has better community support for typical usage.
            t5_model_name = "google/flan-t5-base" 
            t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
            t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
            
            generative_pipeline = pipeline(
                "text2text-generation",
                model=t5_model,
                tokenizer=t5_tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True # Enable sampling for more varied responses
            )
            st.success(f"✅ Loaded Generative model: **{t5_model_name}**")
        except Exception as e:
            st.warning(f"Failed to load generative model ({t5_model_name}). Error: {e}")
            generative_pipeline = None # Ensure it's None if loading fails
            
        return embedding_model, qa_pipeline, generative_pipeline
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def preprocess_text(text):
    """Advanced text preprocessing for better accuracy"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove unwanted characters but keep important punctuation
    # Allowing more punctuation for better sentence parsing
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\'\"]', ' ', text)
    
    # Fix sentence boundaries - look for a period/question/exclamation mark followed by space and uppercase letter
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # Remove very short fragments
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    return ' '.join(sentences)

def extract_text_from_url(url):
    """Extract text from a webpage with better content filtering"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'menu', 'form', 'button', 'noscript']):
            element.decompose()
        
        # Try to find main content areas more reliably
        main_content_tags = ['main', 'article', 'div', 'section']
        main_content = None
        for tag in main_content_tags:
            found = soup.find(tag, {'class': ['content', 'main', 'article', 'post-content', 'entry-content']})
            if found:
                main_content = found
                break
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        
        # Advanced preprocessing
        text = preprocess_text(text)
        
        if len(text) < 100:
            st.warning("⚠️ Very little text extracted. The website might have restrictions, or the content is minimal.")
        
        return text
    except requests.exceptions.RequestException as req_err:
        st.error(f"Network or request error extracting text from URL: {req_err}")
        return None
    except Exception as e:
        st.error(f"Error extracting text from URL: {e}")
        return None

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF with better handling"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        # Advanced preprocessing
        text = preprocess_text(text)
        
        if len(text) < 100:
            st.warning("⚠️ Very little text extracted from PDF. The PDF might be image-based or contain scanned text.")
        
        return text
    except PyPDF2.errors.PdfReadError as pdf_err:
        st.error(f"PDF Read Error: {pdf_err}. The PDF might be corrupted or encrypted.")
        return None
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def intelligent_chunking(text, max_chunk_size=800, overlap=100):
    """Intelligent text chunking that preserves context"""
    if not text:
        return []
    
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = [] # Store sentences, then join
    current_size = 0
    
    for sentence in sentences:
        sentence_words = word_tokenize(sentence)
        sentence_size = len(sentence_words)
        
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk).strip())
            
            # Create overlap: take sentences from the end of the previous chunk
            overlap_sentences = []
            words_count = 0
            # Iterate backwards through sentences in the *just completed* chunk
            for prev_sentence_idx in range(len(current_chunk) -1, -1, -1):
                prev_sentence = current_chunk[prev_sentence_idx]
                if words_count + len(word_tokenize(prev_sentence)) <= overlap:
                    overlap_sentences.insert(0, prev_sentence)
                    words_count += len(word_tokenize(prev_sentence))
                else:
                    break
            
            current_chunk = overlap_sentences + [sentence]
            current_size = len(word_tokenize(" ".join(current_chunk)))
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())
    
    # Filter out very short chunks
    chunks = [chunk for chunk in chunks if len(word_tokenize(chunk)) > 20]
    
    return chunks

def create_embeddings(text_chunks, embedding_model):
    """Create embeddings with better error handling"""
    try:
        if not text_chunks:
            return None
        
        # Process in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        # Use st.progress for visual feedback
        progress_text = "Creating embeddings..."
        embedding_bar = st.progress(0, text=progress_text)

        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            batch_embeddings = embedding_model.encode(
                batch,
                convert_to_tensor=False,
                show_progress_bar=False # Streamlit handles progress bar
            )
            all_embeddings.extend(batch_embeddings)
            progress_val = min(float(i + batch_size) / len(text_chunks), 1.0)
            embedding_bar.progress(progress_val, text=f"{progress_text} {int(progress_val*100)}%")
        
        embedding_bar.empty() # Clear the progress bar after completion
        return np.array(all_embeddings)
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None

def create_faiss_index(embeddings):
    """Create FAISS index with better configuration"""
    try:
        if embeddings is None or len(embeddings) == 0:
            return None
        
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for better similarity search
        index = faiss.IndexFlatIP(dimension)  # Inner product for better similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype('float32'))
        
        index.add(embeddings.astype('float32'))
        return index
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        return None

def find_relevant_chunks(query, embedding_model, index, text_chunks, k=5):
    """Find most relevant chunks with better scoring"""
    try:
        if not query or index is None or not text_chunks:
            return []
        
        # Create query embedding
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)
        faiss.normalize_L2(query_embedding.astype('float32'))
        
        # Search for similar chunks
        # Ensure k doesn't exceed the number of available chunks
        actual_k = min(k, len(text_chunks))
        if actual_k == 0: # No chunks to search
            return []

        scores, indices = index.search(query_embedding.astype('float32'), actual_k)
        
        relevant_chunks = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            # Ensure idx is within bounds and score is meaningful
            if 0 <= idx < len(text_chunks) and score > 0.1:  # Minimum similarity threshold
                relevant_chunks.append({
                    'text': text_chunks[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        # Sort by relevance score
        relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        return relevant_chunks
    except Exception as e:
        st.error(f"Error finding relevant chunks: {e}")
        return []

def generate_comprehensive_answer(question, relevant_chunks, qa_pipeline, generative_pipeline=None):
    """Generate a comprehensive answer using multiple approaches"""
    try:
        if not relevant_chunks:
            return "I couldn't find relevant information in the document to answer your question.", 0.0
        
        # Combine top relevant chunks
        # Limit to 3 chunks to manage context length for models
        context = " ".join([chunk['text'] for chunk in relevant_chunks[:3]])
        
        # Ensure context isn't too long for the QA model
        max_context_length_qa = 512 # Typical max for BERT-like models
        if len(context.split()) > max_context_length_qa:
             # Truncate context by word count to avoid cutting in the middle of a word
            context = " ".join(context.split()[:max_context_length_qa]) + "..."
        
        extractive_answer = None
        confidence = 0.0

        # Try extractive QA first
        try:
            qa_result = qa_pipeline(
                question=question,
                context=context,
                max_answer_len=150, # Max length of the extracted answer
                handle_impossible_answer=True
            )
            
            extractive_answer = qa_result['answer']
            confidence = qa_result['score']
            
            # Filter out generic/empty answers from QA model with low confidence
            if len(extractive_answer.strip()) < 5 or "not find" in extractive_answer.lower():
                extractive_answer = None # Treat as no good answer
                confidence = 0.0

        except Exception as e:
            # print(f"Extractive QA failed for question '{question}': {e}") # For debugging
            extractive_answer = None
            confidence = 0.0
        
        # Try generative approach if available and extractive confidence is low
        generative_answer = None
        # Only use generative if extractive confidence is very low or no extractive answer
        if generative_pipeline and confidence < 0.4: 
            try:
                # Format for T5 model: "summarize: <text>" or "question: <q> context: <c>"
                input_text = f"answer the question: {question} based on the following text: {context}"
                
                # Truncate input for generative model if too long
                max_generative_input_length = 512 # Max tokens for T5 input
                input_tokens = generative_pipeline.tokenizer.encode(input_text, max_length=max_generative_input_length, truncation=True)
                input_text_truncated = generative_pipeline.tokenizer.decode(input_tokens, skip_special_tokens=True)

                gen_result = generative_pipeline(
                    input_text_truncated,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9 # Add top_p for more diverse but coherent answers
                )
                
                generative_answer = gen_result[0]['generated_text']
                # Basic check for empty or unhelpful generative answer
                if not generative_answer or len(generative_answer.strip()) < 10:
                    generative_answer = None

            except Exception as e:
                # print(f"Generative QA failed for question '{question}': {e}") # For debugging
                generative_answer = None
        
        # Choose the best answer
        if extractive_answer and confidence >= 0.4: # Prioritize higher confidence extractive
            final_answer = extractive_answer
            final_confidence = confidence
        elif generative_answer: # If generative provides an answer
            final_answer = generative_answer
            final_confidence = 0.6  # Assign a default moderate-to-high confidence for good generative answers
        elif extractive_answer: # If extractive exists but confidence is lower than 0.4
             final_answer = extractive_answer
             final_confidence = confidence
        else:
            # Fallback: extract most relevant sentences directly
            top_sentences = []
            for chunk in relevant_chunks[:2]: # Look at top 2 chunks for fallback sentences
                sentences = sent_tokenize(chunk['text'])
                for sentence in sentences:
                    # Look for substantial overlap in words, not just single characters
                    question_words = set(word_tokenize(question))
                    sentence_words_lower = set(word_tokenize(sentence.lower()))
                    common_words = question_words.intersection(sentence_words_lower)

                    # Only add if there's significant overlap or it contains a key phrase
                    if len(common_words) > 1 and len(sentence.strip()) > 20: 
                        top_sentences.append(sentence)
                        if len(top_sentences) >= 3:
                            break
                if len(top_sentences) >= 3:
                    break
            
            if top_sentences:
                final_answer = "Based on the document, " + " ".join(top_sentences)
                final_confidence = 0.3 # Lower confidence for simple sentence extraction
            else:
                final_answer = "I couldn't find a direct answer or generate a specific response from the provided document. Here's the most relevant context I found: " + \
                               f"{context[:250]}..." if context else "No relevant context found."
                final_confidence = 0.1 # Very low confidence if no direct answer or generative
        
        return final_answer, final_confidence
        
    except Exception as e:
        st.error(f"An unexpected error occurred during answer generation: {e}")
        return "Sorry, I encountered an internal error while processing your question.", 0.0

def get_confidence_color(confidence):
    """Get color class based on confidence level"""
    if confidence >= 0.6:
        return "confidence-high"
    elif confidence >= 0.3:
        return "confidence-medium"
    else:
        return "confidence-low"

def get_confidence_label(confidence):
    """Get confidence label"""
    if confidence >= 0.6:
        return "High"
    elif confidence >= 0.3:
        return "Medium"
    else:
        return "Low"

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Document Chatbot</h1>
    <p>Upload a PDF or provide a website URL to start chatting with your documents!</p>
    <small>Now with advanced AI models for better accuracy</small>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Document Input")
    
    # Model loading status
    with st.spinner("Loading advanced AI models... (This might take a moment)"):
        embedding_model, qa_pipeline, generative_pipeline = load_models()
    
    if embedding_model and qa_pipeline:
        st.success("✅ Core AI models loaded successfully!")
        if generative_pipeline:
            st.success("✅ Generative AI model also loaded!")
    else:
        st.error("❌ Failed to load essential AI models. Please check your internet connection or try again.")
        # Do not st.stop() here; allow the app to partially run if possible.

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📄 Upload PDF", "🌐 Website URL"],
        key="input_method_radio"
    )
    
    # Document processing
    if input_method == "📄 Upload PDF":
        uploaded_file = st.file_uploader(
            "Upload PDF file",
            type=['pdf'],
            help="Upload a PDF document to chat with",
            key="pdf_uploader"
        )
        
        if uploaded_file is not None:
            if st.button("📊 Process PDF", key="process_pdf_button"):
                with st.spinner("Processing PDF with advanced algorithms..."):
                    text = extract_text_from_pdf(uploaded_file)
                    if text and len(text) > 50:
                        st.session_state.raw_text = text
                        st.session_state.text_chunks = intelligent_chunking(text)
                        
                        if st.session_state.text_chunks:
                            st.session_state.embeddings = create_embeddings(
                                st.session_state.text_chunks, embedding_model
                            )
                            if st.session_state.embeddings is not None:
                                st.session_state.index = create_faiss_index(st.session_state.embeddings)
                                st.session_state.document_processed = True
                                st.success("✅ PDF processed successfully!")
                                st.info(f"📊 Created **{len(st.session_state.text_chunks)}** intelligent chunks.")
                            else:
                                st.error("❌ Failed to create embeddings. This might be due to model loading issues.")
                        else:
                            st.error("❌ No meaningful text chunks could be created from the PDF. It might be empty or image-based.")
                    else:
                        st.error("❌ Could not extract sufficient text from PDF. Please try a different file.")
    
    else:  # Website URL
        url = st.text_input(
            "Enter website URL:",
            placeholder="https://example.com/your-document-page",
            key="url_input"
        )
        
        if st.button("🔍 Process Website", key="process_website_button"):
            if url:
                with st.spinner("Extracting and processing website content..."):
                    text = extract_text_from_url(url)
                    if text and len(text) > 50:
                        st.session_state.raw_text = text
                        st.session_state.text_chunks = intelligent_chunking(text)
                        
                        if st.session_state.text_chunks:
                            st.session_state.embeddings = create_embeddings(
                                st.session_state.text_chunks, embedding_model
                            )
                            if st.session_state.embeddings is not None:
                                st.session_state.index = create_faiss_index(st.session_state.embeddings)
                                st.session_state.document_processed = True
                                st.success("✅ Website processed successfully!")
                                st.info(f"📊 Created **{len(st.session_state.text_chunks)}** intelligent chunks.")
                            else:
                                st.error("❌ Failed to create embeddings. This might be due to model loading issues.")
                        else:
                            st.error("❌ No meaningful text chunks could be created from the website. It might be empty or restricted.")
                    else:
                        st.error("❌ Could not extract sufficient text from website. Please check the URL or try a different site.")
            else:
                st.warning("Please enter a valid URL to process.")
    
    # Document info
    if st.session_state.document_processed:
        st.markdown("### 📈 Document Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(st.session_state.text_chunks)}</h3>
                <p>Smart Chunks</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(st.session_state.raw_text.split())}</h3>
                <p>Total Words</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear document button
        if st.button("🗑️ Clear Document", key="clear_document_button"):
            st.session_state.document_processed = False
            st.session_state.messages = []
            st.session_state.text_chunks = []
            st.session_state.embeddings = None
            st.session_state.index = None
            st.session_state.raw_text = ""
            st.rerun()

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    if st.session_state.document_processed:
        st.markdown("### 💬 Chat with your document")
        
        # Display chat messages in a scrollable container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                confidence_class = get_confidence_color(message.get('confidence', 0))
                confidence_label = get_confidence_label(message.get('confidence', 0))
                
                st.markdown(f"""
                <div class="bot-message">
                    <strong>AI:</strong> {message["content"]}
                    <br><small class="{confidence_class}">Confidence: {confidence_label} ({message.get('confidence', 0):.1%})</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input(
                "Ask a question about your document:",
                placeholder="What is the main topic? Who are the key people mentioned?",
                help="Be specific in your questions for better results"
            )
            
            col_button1, col_button2 = st.columns(2)
            with col_button1:
                submit_button = st.form_submit_button("Send 📤")
            with col_button2:
                if st.form_submit_button("Clear Chat 🧹"):
                    st.session_state.messages = []
                    st.rerun()
        
        if submit_button and user_question:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            with st.spinner("Thinking..."):
                # Check if models are loaded before proceeding
                if embedding_model is None or qa_pipeline is None:
                    st.error("AI models are not loaded. Please try refreshing or check for errors in the sidebar.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Sorry, the AI models are not fully loaded. I cannot answer your question right now.",
                        "confidence": 0.0
                    })
                    st.rerun() # Rerun to show the error message

                # Find relevant chunks
                relevant_chunks = find_relevant_chunks(
                    user_question, 
                    embedding_model, 
                    st.session_state.index, 
                    st.session_state.text_chunks,
                    k=5
                )
                
                # Generate comprehensive answer
                answer, confidence = generate_comprehensive_answer(
                    user_question, 
                    relevant_chunks, 
                    qa_pipeline,
                    generative_pipeline
                )
                
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
            <p><strong>💡 Pro Tips:</strong></p>
            <ul>
                <li>Be specific in your questions</li>
                <li>Use clear, simple language</li>
                <li>Include key terms from document</li>
                <li>Try follow-up questions for more details</li>
                <li>Check confidence scores for reliability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🎯 Advanced Features")
    features = [
        "🧠 Multi-model AI approach (Extractive & Generative)",
        "🔍 Intelligent text chunking for better context",
        "⚡ Semantic similarity search with FAISS",
        "📊 Confidence scoring for answers",
        "🎯 Context-aware answers and fallbacks",
        "🌐 Robust web content extraction",
        "📄 Smart PDF text processing",
        "💬 Conversational memory (though currently limited to current session messages)"
    ]
    
    for feature in features:
        st.markdown(f"• {feature}")
    
    # Model info
    st.markdown("### 🤖 AI Models In Use")
    st.markdown("""
    **Embedding Model:** - `all-mpnet-base-v2`: Excellent for semantic understanding and sentence embeddings.
    
    **Question Answering (QA) Model:**
    - Primary: `deepset/roberta-base-squad2`: A powerful, RoBERTa-based model fine-tuned on SQuAD 2.0 for extractive QA.
    - Fallback: `distilbert-base-cased-distilled-squad`: A smaller, faster model if the primary fails.
    
    **Generative Model (for comprehensive answers):**
    - `google/flan-t5-base`: A versatile and robust text-to-text generation model, capable of summarizing and answering in a conversational style.
    """)
    
    # Tips
    st.markdown("### 💡 Tips for Better Results")
    tips = [
        "Ask **specific** and clear questions.",
        "Use **keywords** directly from the document.",
        "Break down **complex questions** into simpler ones.",
        "If an answer is unclear, try **rephrasing** your question.",
        "Pay attention to the **confidence score** of the answers."
    ]
    
    for tip in tips:
        st.markdown(f"• {tip}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with ❤️ using advanced AI models</p>
    <p>Enhanced with intelligent chunking and multi-model approach</p>
</div>
""", unsafe_allow_html=True)
