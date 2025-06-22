import streamlit as st
import os
import json
from pathlib import Path
from datetime import datetime
from app.communicator.communicator import Communicator

# --- Helper functions ---

def get_prompt_files(directory="data/prompts"):
    """Return list of .txt files in the prompt directory."""
    if not os.path.isdir(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith('.txt')]

def load_prompt_text(filename, directory="data/prompts"):
    """Return the contents of the selected prompt file."""
    path = os.path.join(directory, filename)
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception:
        return ""

def get_download_path(filename=f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_chat-history.json"):
    """Suggest a path in user's Downloads or Documents folder (Windows)."""
    downloads = Path.home() / "Downloads" / filename
    documents = Path.home() / "Documents" / filename
    # Prefer Downloads if it exists, else Documents
    return str(downloads) if downloads.parent.is_dir() else str(documents)

def load_history_from_json(json_str):
    """Load chat history from uploaded JSON string."""
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            # Assume it's just a list of messages
            return data
        elif isinstance(data, dict) and "history" in data:
            # If saved as a dict with 'history' key
            return data["history"]
    except Exception:
        return None
    return None

def clear_input():
    """Clear the user input field."""
    st.session_state["user_input"] = ""

# --- Initialize session state ---

if "communicator" not in st.session_state:
    st.session_state.communicator = Communicator()


# --- Sidebar (settings) ---

st.sidebar.title("Settings")

# System prompt selection
prompt_files = get_prompt_files()
selected_prompt = st.sidebar.selectbox("System Prompt", prompt_files)
base_prompt = load_prompt_text(selected_prompt) if selected_prompt else ""

# Preview system prompt
st.sidebar.write("**Prompt Preview:**")
st.sidebar.code(base_prompt, language="markdown")

# Temperature slider
temperature = st.sidebar.slider(
    "Temperature (creativity)",
    min_value=0.0, max_value=1.0, value=0.35, step=0.01,
    help="Lower = more predictable, higher = more creative. 0.35 is a good value for careful, conscientious psych sparring."
)

# Model selection
models = ["llama3", "mistral", "mixtral", "llama3:70b"]  # Add more as needed later
selected_model = st.sidebar.selectbox("Model", models)

# Upload chat history
st.sidebar.write("**Upload Chat History (.json)**")
uploaded_file = st.sidebar.file_uploader("Choose a chat history file", type=["json"])
if uploaded_file is not None:
    json_bytes = uploaded_file.read()
    try:
        json_str = json_bytes.decode("utf-8")
    except Exception:
        json_str = json_bytes.decode("latin-1")
    loaded_history = load_history_from_json(json_str)
    if loaded_history is not None:
        st.session_state.communicator.history = loaded_history
        st.sidebar.success("Chat history loaded!")
    else:
        st.sidebar.error("Failed to load chat history. Expecting a JSON array or object with 'history' key.")

# Action buttons
if st.sidebar.button("Reset Session"):
    st.session_state.communicator.reset_session()

if st.sidebar.button("Summarise History"):
    summary = st.session_state.communicator.summarize_history()
    st.info(f"Summary:\n\n{summary}")

if st.sidebar.button("Export Session"):
    file_path = get_download_path()
    st.session_state.communicator.export_session(file_path)
    st.success(f"Session exported to: {file_path}")

# --- Main window ---

st.title("PsAIch Sparring Chatbot")

# Display chat history
st.subheader("Chat History")
for msg in st.session_state.communicator.history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "assistant":
        # st.markdown(f"**AI:** {msg['content']}")
        st.markdown(f"**{selected_model}:** {msg['content']}")
    elif msg["role"] == "system":
        st.markdown(f"**System prompt:** {msg['content']}")

# Chat input
st.subheader("Your Message")
user_input = st.text_area("Type your message...", height=100)

if st.button("Send", on_click=clear_input) and user_input.strip():
    # Set Communicator params
    st.session_state.communicator.model_name = selected_model
    st.session_state.communicator.temperature = temperature

    # If chat history is too long, summarize before answering
    context = st.session_state.communicator.get_context()
    if "Warning: context length exceeded" in context or "summary" in context.lower():
        # Already summarized inside get_context, just proceed
        pass

    # Send message using Communicator
    response = st.session_state.communicator.send_to_llm(
        prompt=user_input,
        system_prompt=base_prompt
    )
    st.write("Assistant response:", response)
    st.rerun()  # Refresh to show updated chat history