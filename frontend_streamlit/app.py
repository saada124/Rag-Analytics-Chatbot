import streamlit as st
import requests
import json
import os

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
    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }
    
    /* Header styling */
    h1 {
        color: #1E3A8A;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 10px 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 15px;
        border: 1px solid #e5e7eb;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        color: #f8fafc;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f8fafc;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #10b981;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
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
                st.dataframe(message["data"], use_container_width=True)

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
                            st.dataframe(analytics_data, use_container_width=True)
                            
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
