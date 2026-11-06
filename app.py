
import os
import uuid
import tempfile
import streamlit as st
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from backend import chatbot, get_all_threads, ingest_rag_document

# Helper functions
def generate_thread_id():
    return str(uuid.uuid4())

def get_config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def text_of(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text = ""
        for item in content:
            if isinstance(item, str):
                text += item
            elif isinstance(item, dict) and item.get("type") == "text":
                text += item.get("text", "")
        return text

    return str(content) if content else ""


def load_history(thread_id):
    state = chatbot.get_state(get_config(thread_id))
    history = []

    for message in state.values.get("messages", []):
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        text = text_of(message.content)

        if text.strip():
            history.append({"role": role, "content": text})

    return history


def get_pending_plan(thread_id):

    config = get_config(thread_id)

    try:
        snapshot = chatbot.get_state(config, subgraphs=True)

    except Exception:
        snapshot = chatbot.get_state(config)

    for task in snapshot.tasks:
        if task.interrupts:

            payload = dict(task.interrupts[-1].value)

            sub_state = getattr(task, "state", None)

            if sub_state and sub_state.values.get("plan"):
                payload["plan"] = sub_state.values["plan"]

            return payload

    return None


#Run LangGraph and stream the response

def stream_graph(graph_input, thread_id, status_holder):

    for chunk, metadata in chatbot.stream(
        graph_input,
        config=get_config(thread_id),
        stream_mode="messages",
    ):
        if isinstance(chunk, ToolMessage):
            tool_name = chunk.name or "tool"
            label = f"Using {tool_name}..."

            if status_holder["box"] is None:
                status_holder["box"] = st.status(label, expanded=True)
            else:
                status_holder["box"].update(
                    label=label,
                    state="running",
                )

        elif (
            isinstance(chunk, AIMessage)
            and metadata.get("langgraph_node") == "chat_node"
            and chunk.content
        ):
            yield text_of(chunk.content)


def invoke_llm(graph_input, thread_id, busy_label="Thinking..."):

    status_holder = {"box": None}

    with st.chat_message("assistant"):
        with st.spinner(busy_label):
            reply = st.write_stream(
                stream_graph(graph_input, thread_id, status_holder)
            )

        if status_holder["box"] is not None:
            pending = get_pending_plan(thread_id)


            if pending:
                label = "⏸️ Waiting for your review"
            else:
                label = "✅ Tool finished"

            status_holder["box"].update(
                label=label,
                state="complete",
                expanded=False,
            )

    if reply:
        st.session_state.message_history.append({
            "role": "assistant",
            "content": reply,
        })

    st.rerun()


# Set up the page

st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖")
st.title("Agentic Chatbot")

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()


if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = get_all_threads()

thread_id = st.session_state.thread_id

if thread_id not in st.session_state.chat_threads:
    st.session_state.chat_threads.append(thread_id)


#Sidebar: select or create a chat

st.sidebar.title("My Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    new_id = generate_thread_id()

    st.session_state.thread_id = new_id
    st.session_state.message_history = []
    st.session_state.chat_threads.append(new_id)
    

for tid in reversed(st.session_state.chat_threads):
    history = load_history(tid)

    title = "New Chat"

    for message in history:
        if message["role"] == "user":
            title = message["content"][:30]
            if len(message["content"]) > 30:
                title += "..."
            break

    if title == "New Chat":
        continue

    if st.sidebar.button(
        title,
        key=f"thread_{tid}",
        type="primary" if tid == thread_id else "secondary",
        use_container_width=True,
    ):
        st.session_state.thread_id = tid
        st.session_state.message_history = history
        


#Display chat history

for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Human review

pending = get_pending_plan(thread_id)

if pending:
    st.warning("🧑 Human review required")
    st.write("Task:", pending["task"])
    st.markdown(text_of(pending["plan"]))

    with st.form(key=f"review_{thread_id}"):
        decision = st.radio(
            "Choose an action",
            ["Approve", "Revise", "Reject"],
            horizontal=True,
        )

        feedback = st.text_area(
            "Feedback",
            placeholder="Explain what you want to change...",
        )

        submitted = st.form_submit_button("Submit decision")

    if submitted:
        if decision == "Revise" and not feedback.strip():
            st.warning("Please enter feedback.")
        else:
            invoke_llm(
                Command(resume={
                    "decision": decision.lower(),
                    "feedback": feedback if decision == "Revise" else "",
                }),
                thread_id,
                "Revising the plan..." if decision == "Revise"
                else "Working on it...",
            )


# Chat input and PDF upload

submission = st.chat_input(
    "Review the plan above first..." if pending else "Type here",
    accept_file=True,
    file_type=["pdf", "txt", "csv"],
    disabled=pending is not None,
)


if submission:
    user_input = submission.text
    uploaded_file = (
            submission.files[0]
            if submission.files
            else None
        )
    if submission.files:

        filename = uploaded_file.name
        extension = os.path.splitext(filename)[1].lower()
        temp_path = None
        upload_success = False

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension,
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            with st.spinner(f"Processing {filename}..."):
                ingest_rag_document(
                    temp_path,
                    thread_id,
                    filename,
                )

            upload_success = True
            st.toast(f"{filename} processed successfully.")

        except Exception as error:
            st.error(f"File processing failed: {error}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        if not upload_success:
            st.stop()

    if user_input or uploaded_file is not None:
        file_name = uploaded_file.name if uploaded_file else None
        display_text = (
                    user_input
                    if user_input
                    else f"Please review the {file_name} file."
                )

        st.session_state.message_history.append({
            "role": "user",
            "content": display_text,
            "file_name": file_name
        })

        with st.chat_message("user"):
            st.markdown(display_text)
        invoke_llm(
            {"messages": [HumanMessage(content=display_text)]},
            thread_id,
        )
