import streamlit as st
from rag_pipeline import (
    BASE_DIR,
    CHROMA_PATH,
    generate_answer,
    get_pdf_collection,
    is_slc_question,
    page_link,
    retrieve_context,
    validate_api_key,
)


INGEST_SCRIPT_PATH = BASE_DIR / "ingest_pdfs.py"

st.set_page_config(
    page_title="SLC Student Finance Assistant",
    page_icon="SLC",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"] { background: #f4f7f4; }
    [data-testid="stSidebar"] { background: #173b32; }
    [data-testid="stSidebar"] * { color: #f5f7f3; }
    [data-testid="stSidebar"] input {
        color: #14221d !important;
        background: #ffffff !important;
        border-color: #a9c3b5 !important;
    }
    [data-testid="stSidebar"] input::placeholder {
        color: #5d7066 !important;
        opacity: 1;
    }
    [data-testid="stAppScrollToBottomContainer"] {
        background: #f4f7f4 !important;
        color: #18352c !important;
        --text-color: #18352c;
        --background-color: #f4f7f4;
        --secondary-background-color: #e7eee9;
        --primary-color: #a84f24;
    }
    [data-testid="stAppScrollToBottomContainer"] :is(h1, h2, h3, h4, p, label, span, small, strong, li) {
        color: #18352c !important;
    }
    .main-title { color: #173b32; font-family: Georgia, serif; margin-bottom: 0; }
    .main-subtitle { color: #52665d; margin-top: 0.35rem; }
    [data-testid="stAppScrollToBottomContainer"] [data-testid="stChatMessage"] {
        background: #e7eee9;
        border: 1px solid #d2dfd7;
        border-radius: 8px;
    }
    [data-testid="stAppScrollToBottomContainer"] [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        color: #18352c !important;
    }
    [data-testid="stAppScrollToBottomContainer"] [data-testid="stChatInput"] textarea::placeholder {
        color: #5d7066 !important;
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pdf_collection():
    if not CHROMA_PATH.exists():
        return None
    return get_pdf_collection()


def clear_api_validation() -> None:
    st.session_state.api_key_validated = False
    st.session_state.validated_api_key = ""


if "api_key_validated" not in st.session_state:
    st.session_state.api_key_validated = False
    st.session_state.validated_api_key = ""
st.session_state.pop("openrouter_api_key", None)
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.markdown("## SLC Assistant")
    st.caption("Connect to OpenRouter to search the indexed SLC PDF guidance.")
    st.text_input(
        "OpenRouter API key",
        type="password",
        key="openrouter_api_key_input_v2",
        on_change=clear_api_validation,
        help="Used for OpenRouter embeddings and chat completions. It is kept in this Streamlit session.",
    )
    if st.button("Validate key", use_container_width=True):
        try:
            validate_api_key(st.session_state.openrouter_api_key_input_v2)
        except Exception as error:
            clear_api_validation()
            st.error(f"Validation failed ({type(error).__name__}). Check the key and model access.")
        else:
            st.session_state.api_key_validated = True
            st.session_state.validated_api_key = st.session_state.openrouter_api_key_input_v2.strip()
            st.success("Key validated. Chat is enabled.")

    if st.session_state.api_key_validated:
        st.caption("Connected to OpenRouter")
    else:
        st.caption("Validate your key to enable the chatbot.")


collection = load_pdf_collection()
collection_count = collection.count() if collection is not None else 0

st.markdown('<h1 class="main-title">Student Finance - SLC guidance</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="main-subtitle">Ask about eligibility, applications, loans, repayments, DSA, and other indexed SLC guidance.</p>',
    unsafe_allow_html=True,
)

if not INGEST_SCRIPT_PATH.is_file():
    st.error("The PDF ingestion script was not found beside this app.")
elif collection is None:
    st.warning("The PDF index was not found. Run `python ingest_pdfs.py` first.")
elif collection_count == 0:
    st.warning("The SLC PDF collection is empty. Run `python ingest_pdfs.py` first.")
else:
    st.caption(f"Searching {collection_count:,} indexed PDF passages")

if not st.session_state.api_key_validated:
    st.info("Enter and validate your OpenRouter key in the left panel to start chatting.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    link = page_link(source["url"], source["page"])
                    label = f"PDF page {source['page']}" if source["page"] else "PDF source"
                    if link:
                        st.markdown(f"[{label}]({link})")
                    else:
                        st.write(label)

chat_disabled = not st.session_state.api_key_validated or collection_count == 0
question = st.chat_input(
    "Ask an SLC student finance question...",
    disabled=chat_disabled,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    answer = ""
    sources = []
    with st.chat_message("assistant"):
        with st.spinner("Checking the question and searching SLC guidance..."):
            try:
                api_key = st.session_state.validated_api_key
                passages, sources = retrieve_context(api_key, collection, question)
                if not passages:
                    answer = "I could not find relevant text in the indexed SLC PDF documents."
                    sources = []
                elif not is_slc_question(api_key, question, passages):
                    answer = "I can only help with questions about Student Loans Company (SLC) services and UK student finance guidance."
                    sources = []
                else:
                    answer = generate_answer(api_key, question, passages)
            except Exception as error:
                answer = "I couldn't complete that request. Check the OpenRouter key, model access, and local Chroma index, then try again."
                st.error(f"Request failed ({type(error).__name__}).")

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    link = page_link(source["url"], source["page"])
                    label = f"PDF page {source['page']}" if source["page"] else "PDF source"
                    if link:
                        st.markdown(f"[{label}]({link})")
                    else:
                        st.write(label)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
