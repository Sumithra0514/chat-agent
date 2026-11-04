import os
import ast
import math
import operator
import sqlite3
import requests
from typing import Annotated, Any, TypedDict
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from typing_extensions import NotRequired

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader
)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


from dotenv import load_dotenv
load_dotenv()


llm = ChatOpenAI(model="gpt-5-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

FAISS_ROOT = "faiss_db"

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: NotRequired[str]
    plan: NotRequired[str]
    feedback: NotRequired[str]
    decision: NotRequired[str]
    planner_tool_call_id: NotRequired[str]


@tool
def task_planner(task: str) -> str:
    """Create a five-step plan and request human approval,
    revision, or rejection."""
    return task


def generate_plan(state: ChatState):
    response = llm.invoke(
        [
                  SystemMessage(content="""
        You are a planning assistant.
        Create a plan based on the user's task.
        When revising, modify the previous plan according
        to the user's feedback instead of starting over.
        """),
        HumanMessage(content=f"""
        Task: {state['task']}

        Previous plan:
        {state.get('plan', '')}

        Revision feedback:
        {state.get('feedback', '')}

        Generate the revised plan.
        """)
            ,
        ]
    )
    
    return {"plan": response.content}


def human_review(state: ChatState):
    answer = interrupt(
        {
            "type": "task_plan_review",
            "task": state["task"],
            "plan": state["plan"],
            "message": "Review this plan.",
            "options": ["approve", "revise", "reject"],
        }
    )
    return {
        "decision": answer["decision"].lower(),
        "feedback": answer.get("feedback", ""),
    }


def decide(state: ChatState):
    if state["decision"] == "approve":
        return "proceed"
    if state["decision"] == "revise":
        return "generate_plan"
    return "reject"


def proceed(state: ChatState):
    return {
        "messages": [
            ToolMessage(
                content=f"Human-approved plan:\n{state['plan']}",
                tool_call_id=state["planner_tool_call_id"],
            )
        ]
    }


def reject_plan(state: ChatState):
    return {
        "messages": [
            ToolMessage(
                content="The human rejected the task plan.",
                tool_call_id=state["planner_tool_call_id"],
            )
        ]
    }


planner_builder = StateGraph(ChatState)
planner_builder.add_node("generate_plan", generate_plan)
planner_builder.add_node("human_review", human_review)
planner_builder.add_node("proceed", proceed)
planner_builder.add_node("reject_plan", reject_plan)

planner_builder.add_edge(START, "generate_plan")
planner_builder.add_edge("generate_plan", "human_review")
planner_builder.add_conditional_edges(
    "human_review",
    decide,
    {"proceed": "proceed", "generate_plan": "generate_plan", "reject": "reject_plan"},
)
planner_builder.add_edge("proceed", END)
planner_builder.add_edge("reject_plan", END)

planner_graph = planner_builder.compile()


def _index_path(thread_id: str) -> str:
    return os.path.join(FAISS_ROOT, thread_id)


def ingest_rag_document(
        file_path: str, 
        thread_id: str, 
        filename: str) -> None:
    
    extension = os.path.splitext(filename)[1].lower()

    if extension == ".pdf":
        docs = PyPDFLoader(file_path).load()

    elif extension == ".txt":
        docs = TextLoader(
            file_path,
            encoding="utf-8",
        ).load()

    elif extension == ".csv":
        docs = CSVLoader(file_path).load()

    else:
        raise ValueError(f"Unsupported file type: {extension}")

    for doc in docs:
        doc.metadata["source"] = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No text content found.")

    path = _index_path(thread_id)
    os.makedirs(FAISS_ROOT, exist_ok=True)

    if os.path.exists(path):
        vector_store = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(path)


def get_retriever(thread_id: str):
    path = _index_path(thread_id)
    if not os.path.exists(path):
        return None
    
    vector_store = FAISS.load_local(
        folder_path=path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


@tool
def rag_tool(query: str, config: RunnableConfig) -> str:

    thread_id = config.get("configurable", {}).get("thread_id")
    retriever = get_retriever(thread_id) if thread_id else None

    if retriever is None:
        return "No file has been uploaded for this conversation yet."

    documents = retriever.invoke(query)
    if not documents:
        return "No relevant information was found in the file."

    formatted = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")
        formatted.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )
    return "\n\n".join(formatted)



search_tool = TavilySearch(max_results=5, topic="general", search_depth="advanced")


def _safe_pow(base, exp):
    if abs(exp) > 100 or abs(base) > 1e6:
        raise ValueError("Exponent or base too large")
    return operator.pow(base, exp)


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):

        if node.value.id == "math" and node.attr in _CONSTS:
            return _CONSTS[node.attr]
    if isinstance(node, ast.Call) and not node.keywords:
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "math"
        ):
            name = func.attr
        if name in _FUNCS:
            return _FUNCS[name](*[_eval_node(a) for a in node.args])
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return str(_eval_node(tree.body))
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_stock_price(symbol: str) -> dict:

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "Set the ALPHAVANTAGE_API_KEY environment variable."}

    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as error:
        return {"error": f"Could not fetch stock price: {error}"}


@tool
def get_current_weather(location: str) -> str:
 
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return (
            "Weather API key is missing. "
            "Set the OPENWEATHER_API_KEY environment variable."
        )

    try:
        geo_response = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": location, "limit": 1, "appid": api_key},
            timeout=10,
        )
        geo_response.raise_for_status()
        locations: list[dict[str, Any]] = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        place = locations[0]
        resolved_name = place.get("name", location)
        country = place.get("country", "")
        state = place.get("state", "")

        weather_response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": place["lat"],
                "lon": place["lon"],
                "appid": api_key,
                "units": "metric",
            },
            timeout=10,
        )
        weather_response.raise_for_status()
        data = weather_response.json()

        visibility_m = data.get("visibility")
        visibility_km = round(visibility_m / 1000, 1) if visibility_m is not None else "N/A"
        display_location = ", ".join(p for p in [resolved_name, state, country] if p)
        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {data['weather'][0]['description'].title()}\n"
            f"- Temperature: {data['main']['temp']}°C\n"
            f"- Feels like: {data['main']['feels_like']}°C\n"
            f"- Humidity: {data['main']['humidity']}%\n"
            f"- Pressure: {data['main']['pressure']} hPa\n"
            f"- Wind speed: {data.get('wind', {}).get('speed', 'N/A')} m/s\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "The weather service request timed out. Please try again."
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else "unknown"
        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."
        return f"Weather API returned an HTTP error: {status_code}"
    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"
    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"



tools = [search_tool, calculator, get_stock_price, get_current_weather, rag_tool]
all_llm_tools = tools + [task_planner]
llm_with_tools = llm.bind_tools(all_llm_tools, parallel_tool_calls=False)
tool_node = ToolNode(tools)

SYSTEM_PROMPT = (
    "You are a helpful Agentic Chatbot with access to several tools.\n\n"
    "Tool usage instructions:\n"
    "- Use `rag_tool` for questions about uploaded PDF, TXT, and CSV files. "
    "Always retrieve relevant document content before answering questions about these files.\n"
    "- Use `search_tool` for current events, recent information, or information "
    "that requires an internet search.\n"
    "- Use `calculator` for mathematical calculations. Do not calculate complex "
    "expressions manually when the calculator is available.\n"
    "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
    "- Use `get_current_weather` when the user asks for current weather at a location.\n"
    "- Use `task_planner` whenever the user asks to create a plan or break a goal "
    "into steps. Pass the task description to `task_planner`. "
    "Do not create the plan directly; the planner workflow must generate the "
    "plan and request human approval.\n\n"
    "Answer general questions directly when no tool is required. "
    "Do not invent information from uploaded documents. "
    "If the user asks about a file but no document is available, ask them to upload it. "
    "After receiving a tool result, provide a clear and helpful final answer."
)


def chat_node(state: ChatState):
    """LLM node that can answer directly or call an appropriate tool."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def route_after_chat(state: ChatState):


    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return "end"

    if any(call["name"] == "task_planner" for call in last_message.tool_calls):
        return "prepare_planner"

    return "tools"


def prepare_planner(state: ChatState):
    last_message = state["messages"][-1]
    call = next(c for c in last_message.tool_calls if c["name"] == "task_planner")
    

    return {
        "task": call["args"]["task"],
        "plan": "",
        "feedback": "",
        "decision": "",
        "planner_tool_call_id": call["id"],
    }


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)
checkpoint.setup()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_node("prepare_planner", prepare_planner)
graph.add_node("planner_graph", planner_graph)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    route_after_chat,
    {"tools": "tools", "prepare_planner": "prepare_planner", "end": END},
)
graph.add_edge("prepare_planner", "planner_graph")
graph.add_edge("planner_graph", "chat_node")
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads():
    threads = []

    for checkpoint_tuple in checkpoint.list(None):
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]

        if thread_id not in threads:
            threads.append(thread_id)

    return threads[::-1]