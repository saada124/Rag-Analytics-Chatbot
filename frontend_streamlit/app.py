import streamlit as st
import requests
import json
import os
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Vilavi Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global settings */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main background with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #f0f4fd 0%, #e1e9fb 100%);
    }
    
    /* Header styling with gradient text */
    h1 {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    
    /* Sleek light Glassmorphism for sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.6);
    }
    
    /* Make sure sidebar titles/text look good on the light background */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: #1e293b;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(30, 41, 59, 0.1);
    }
    
    /* Chat message container animations & styling */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.8);
        animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        opacity: 0;
        transform: translateY(10px);
    }
    
    /* Distinct styling for assistant vs user messages */
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-right: 4px solid #3B82F6;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: linear-gradient(135deg, #ffffff 0%, #fdfdff 100%);
        border-left: 4px solid #10b981;
    }

    @keyframes fadeIn {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #34d399 !important;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(52, 211, 153, 0.4);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.39);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.23);
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%);
    }
    
    /* Input box styling */
    .stChatInputContainer {
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.8);
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: transparent;
        color: #475569;
        font-weight: 500;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Backend Connection Check
@st.cache_data(ttl=10)
def check_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None

def clear_memory():
    try:
        requests.post(f"{API_URL}/clear-memory", timeout=5)
        st.session_state.messages = []
        st.toast("Conversation memory cleared successfully!", icon="✅")
    except Exception as e:
        st.error(f"Failed to clear memory on the server: {e}")

# Sidebar
with st.sidebar:
    st.title("🤖 Vilavi Chatbot")
    st.markdown("Your intelligent assistant for Analytics and Documentation.")
    
    st.divider()
    
    st.subheader("System Status")
    health_data = check_health()
    if health_data and health_data.get("status") == "ok":
        st.success("Connected to Backend")
        if "vector_documents" in health_data:
            st.metric("Documents Indexed", health_data["vector_documents"])
    else:
        st.error("Backend Disconnected")
        st.caption("Please ensure the FastAPI server is running.")
        
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        clear_memory()

# Main Application
st.title("Vilavi Assistant")
st.markdown("Ask me questions about enterprise policies, documentations, or analytics data!")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if any
        if message.get("sources"):
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.caption(f"- {source}")
                    
        # Display data payload if analytics
        if message.get("data") and len(message["data"]) > 0:
            with st.expander("View Data Records"):
                st.dataframe(pd.DataFrame(message["data"]).astype(str), use_container_width=True)

# Chat Input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Backend API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if not health_data:
            st.error("Cannot reach the backend server. Please check the connection.")
            st.stop()
            
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"message": prompt},
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    sources = data.get("sources", [])
                    analytics_data = data.get("data", [])
                    if not analytics_data and data.get("analytics"):
                        analytics_data = data["analytics"].get("data", [])
                    
                    message_placeholder.markdown(answer)
                    
                    if sources:
                        with st.expander("View Sources"):
                            for source in sources:
                                st.caption(f"- {source}")
                                
                    if analytics_data and len(analytics_data) > 0:
                        with st.expander("View Data Records"):
                            st.dataframe(pd.DataFrame(analytics_data).astype(str), use_container_width=True)
                            
                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources,
                        "data": analytics_data
                    })
                else:
                    error_msg = f"Error from server: {response.status_code}"
                    message_placeholder.error(error_msg)
            except Exception as e:
                message_placeholder.error(f"An error occurred: {e}")
