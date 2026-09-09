import requests
import streamlit as st

from guardrails import validate_query, check_safety, validate_response

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="OmniBrain",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------
# Session State
# ---------------------------------------------------------

if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def login(api_url: str, username: str, password: str):
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/token",
            data={"username": username, "password": password},
            timeout=15,
        )
    except requests.RequestException as exc:
        return None, f"Could not connect to the API: {exc}"

    if response.ok:
        return response.json().get("access_token"), None

    try:
        detail = response.json().get("detail", "Login failed")
    except ValueError:
        detail = f"Login failed (HTTP {response.status_code})"
    return None, str(detail)


def ask_query(api_url: str, token: str, text: str, dataset_id):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"text": text}
    if dataset_id is not None:
        payload["dataset_id"] = dataset_id

    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/query/",
            json=payload,
            headers=headers,
            timeout=60,
        )
    except requests.RequestException as exc:
        return None, f"Could not connect to the API: {exc}"

    if response.ok:
        return response.json(), None

    try:
        detail = response.json().get("detail", "Query failed")
    except ValueError:
        detail = f"Query failed (HTTP {response.status_code})"
    return None, str(detail)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🧠 OmniBrain")
st.caption("Conversational interface for querying your data")

# ---------------------------------------------------------
# Sidebar: API Login
# ---------------------------------------------------------

with st.sidebar:
    st.header("Backend Connection")
    api_url = st.text_input("FastAPI URL", value="http://127.0.0.1:8000")

    if st.session_state.access_token:
        st.success("Logged in")
        if st.button("Log out", use_container_width=True):
            st.session_state.access_token = None
            st.session_state.chat_history = []
            st.rerun()
    else:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Log in", type="primary", use_container_width=True):
            if not username or not password:
                st.error("Enter both username and password.")
            else:
                token, error = login(api_url, username, password)
                if error:
                    st.error(error)
                else:
                    st.session_state.access_token = token
                    st.success("Login successful")
                    st.rerun()

    st.divider()
    st.subheader("Dataset")
    dataset_id_text = st.text_input(
        "Dataset ID (optional)",
        placeholder="e.g. 1",
        help="Enter an uploaded dataset ID to ask questions about that dataset. Leave blank for a general query.",
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------
# Chat UI
# ---------------------------------------------------------

st.subheader("Chat")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.access_token:
    st.info("Log in from the sidebar to start chatting with OmniBrain.")
else:
    prompt = st.chat_input("Ask a question about your data...")

    if prompt:
        prompt = prompt.strip()
        if not prompt:
            st.warning("Please enter a question.")
        else:
            # Stop invalid or unsafe prompts before they reach FastAPI.
            is_valid, validation_message = validate_query(prompt)
            if not is_valid:
                st.error(validation_message)
                st.stop()

            is_safe, safety_message = check_safety(prompt)
            if not is_safe:
                st.error(safety_message)
                st.stop()

            dataset_id = None
            if dataset_id_text.strip():
                try:
                    dataset_id = int(dataset_id_text.strip())
                    if dataset_id < 1:
                        raise ValueError
                except ValueError:
                    st.error("Dataset ID must be a positive integer.")
                    st.stop()

            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result, error = ask_query(
                        api_url,
                        st.session_state.access_token,
                        prompt,
                        dataset_id,
                    )

                if error:
                    st.error(error)
                    assistant_text = f"**Error:** {error}"
                else:
                    assistant_text = str(result.get("answer", ""))
                    output_valid, output_message = validate_response(assistant_text)

                    if not output_valid:
                        assistant_text = f"**Error:** {output_message}"
                        st.error(output_message)
                    else:
                        assistant_text = output_message
                        st.markdown(assistant_text)

                    metadata = []
                    if result.get("dataset_id") is not None:
                        metadata.append(f"Dataset ID: {result['dataset_id']}")
                    if result.get("rows_analyzed") is not None:
                        metadata.append(f"Rows analyzed: {result['rows_analyzed']}")
                    if result.get("columns_analyzed") is not None:
                        metadata.append(f"Columns analyzed: {result['columns_analyzed']}")
                    if metadata:
                        st.caption(" • ".join(metadata))

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": assistant_text}
                )

# ---------------------------------------------------------
# Image Display (existing Module 2 work)
# ---------------------------------------------------------

st.divider()
st.subheader("🖼️ Image Display")
st.write("Upload an image in JPG, JPEG, or PNG format.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info("Please select an image to continue.")
elif uploaded_file.size == 0:
    st.error("The uploaded file is empty.")
else:
    st.success("Image uploaded successfully!")
    st.write(f"**File name:** {uploaded_file.name}")
    st.write(f"**File type:** {uploaded_file.type}")
    st.write(f"**File size:** {uploaded_file.size / 1024:.2f} KB")
    st.image(uploaded_file, caption=uploaded_file.name, width=600)
