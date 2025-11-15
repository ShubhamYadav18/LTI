import streamlit as st
import requests
from PIL import Image

# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000/ask/"

# Streamlit App UI
st.set_page_config(page_title="SOC Analyst Assistant", page_icon="💬", layout="wide")

# Header with image (optional)
st.title("💡 Incident RAG Investigation System")
st.markdown("""
    **Welcome to your virtual IT support assistant!**
    Please enter your **User ID** and *Incident-related question** below to get instant support.
""")

# Image (Optional)
# Uncomment if you want to display an image
# image = Image.open("path_to_image.jpg")
# st.image(image, caption="IT Support Chatbot")

# Input fields for user ID and query
st.sidebar.header("Enter your Details")
user_id = st.sidebar.text_input("Enter your User ID", placeholder="e.g., user123")
query = st.sidebar.text_area("Enter your Question", placeholder="Describe your issue...")

# Button to clear chat history
if st.sidebar.button("Clear History"):
    if user_id in st.session_state.chat_history:
        del st.session_state.chat_history[user_id]
        st.success("History cleared for this session!")

# Session state to store chat history (per user)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}

# Initialize history for the user
if user_id and user_id not in st.session_state.chat_history:
    st.session_state.chat_history[user_id] = []

# Display previous conversation history
if user_id in st.session_state.chat_history:
    st.subheader("🗨️ Conversation History")
    for i, (q, a) in enumerate(st.session_state.chat_history[user_id]):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a}")
        st.write("---")

# Button to send the query
if st.button("Ask"):
    if user_id and query:
        # Send a POST request to the FastAPI backend with the user ID and query
        try:
            # Show buffering icon while processing
            with st.spinner('Analyzing your request...'):
                # Prepare the data
                payload = {"user_id": user_id, "query": query}
                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    # Parse and display the answer from the response
                    result = response.json()
                    answer = result.get("answer")
                    
                    # Display answer with more styling
                    st.markdown(f"### 📝 **Answer:**")
                    st.write(answer)

                    # Update chat history in session state
                    st.session_state.chat_history[user_id].append((query, answer))

                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter both User ID and a Question.")

# Styling with custom CSS
st.markdown("""
    <style>
        .stTextInput, .stTextArea {
            background-color: #f1f1f1;
            border-radius: 8px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 12px 20px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stSidebar>div {
            background-color: #f5f5f5;
            padding: 20px;
        }
        .stMarkdown {
            font-size: 16px;
            line-height: 1.5;
            color: #333333;
        }
        .stSuccess {
            background-color: #d4edda;
            color: #155724;
        }
        .stError {
            background-color: #f8d7da;
            color: #721c24;
        }
    </style>
""", unsafe_allow_html=True)

# Display a footer message
st.markdown("""
    <footer style="text-align:center; margin-top: 50px;">
        <small>Powered by Streamlit, FastAPI, and OpenAI's GPT</small>
    </footer>
""", unsafe_allow_html=True)
