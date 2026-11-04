# 🤖 Agentic AI Chatbot

An Agentic AI Chatbot built using **LangChain, LangGraph, and OpenAI**. It can understand user queries, select and execute tools, retrieve information from uploaded documents, and generate task plans with human approval.

## ✨ Features

- **Agentic AI Workflow:** Uses LangGraph to manage the chatbot's decision-making and execution flow.
- **Tool Calling:** Automatically selects and executes appropriate tools based on user queries.
- **RAG (Retrieval-Augmented Generation):** Upload PDF, TXT, and CSV documents and ask questions based on their content.
- **FAISS Vector Search:** Stores document embeddings locally and retrieves relevant information using semantic similarity search.
- **Web Search:** Uses Tavily to retrieve information from the web.f
- **Calculator:** Performs mathematical calculations.
- **Weather Information:** Retrieves current weather information using OPENWEATHER API.
- **Stock Information:** Retrieves stock prices using  Alpha Vantage API.
- **Human-in-the-Loop (HITL):** Allows users to review, approve, revise, or reject AI-generated task plans before proceeding.
- **Conversation Memory:** Maintains separate conversation threads using LangGraph and SQLite.

## Deployment

The application has been deployed using **Render** as separate frontend and backend services.

- Frontend: Streamlit
- Hosting platform: Render

**Demo URL:** https://chat-agent-0p37.onrender.com

*Note: The application is currently suspended to manage hosting resources. It can be reactivated for a demonstration upon request.*

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **AI Workflow:** LangGraph
- **LLM:** OpenAI 
- **LLM Framework:** LangChain
- **Tool Calling:** LangChain Tool Calling, LangGraph 
- **Human-in-the-Loop:** LangGraph
- **Embeddings:** OpenAI 
- **Vector Database:** FAISS
- **Document Processing:** LangChain Document Loaders, RecursiveCharacterTextSplitter
- **Web Search:** Tavily API
- **Weather API:** OPENWEATHER API
- **Stock Market API:** Alpha Vantage API
- **Conversation Persistence:** SQLite, LangGraph SQLite Checkpointer
- **Language:** Python


## 🔧 Available Tools

- **Tavily Search Tool:** Searches the web for information.
- **RAG Tool:** Retrieves relevant content from uploaded documents.
- **Calculator:** Performs mathematical calculations.
- **Weather Tool:** Retrieves current weather information.
- **Stock Price Tool:** Retrieves stock market prices.
- **Task Planner:** Generates task plans and starts the human review workflow.

## 🧠 Agentic Workflow

The chatbot uses LangGraph to decide whether a user query requires a tool or can be answered directly.

1. User sends a message.
2. The LLM analyzes the query.
3. The LLM either generates a direct response or selects an appropriate tool.
4. LangGraph executes the selected tool.
5. The tool result is passed back to the LLM.
6. The LLM generates the final response.

## 📚 Document RAG Workflow

The chatbot supports document-based question answering using Retrieval-Augmented Generation (RAG).

1. Upload a PDF, TXT, or CSV file.
2. The appropriate LangChain document loader extracts the content.
3. `RecursiveCharacterTextSplitter` divides documents into chunks.
4. Each chunk is converted into an embedding using OpenAI `text-embedding-3-small`.
5. Embeddings are stored in a local FAISS index for each conversation thread.
6. When the user asks a question about the document, the RAG tool retrieves the four most relevant chunks.
7. The retrieved information is passed to the LLM to generate an answer.


## 👤 Human-in-the-Loop (HITL)

The chatbot implements a human approval workflow using LangGraph.

When a user requests a task plan, the chatbot generates a plan and pauses execution using LangGraph's `interrupt()` function.

The generated plan is displayed in the Streamlit interface, allowing the user to choose:

- **Approve:** Accept the generated plan and continue the workflow.
- **Revise:** Provide feedback and ask the LLM to generate an updated plan.
- **Reject:** Reject the generated plan.

The workflow resumes based on the user's decision using `Command(resume=...)`.

This provides human oversight and control over AI-generated task plans.

## 💾 Conversation Persistence

- Uses SQLite with LangGraph's `SqliteSaver` checkpointer.
- Assigns a unique thread ID to each conversation.
- Saves conversation state and checkpoints.
- Allows users to create new conversations and switch between existing threads.
- Restores conversation history from saved graph state.



## Project Structure

```text
chat-agent/
│
├── app.py
├── backend.py
├── requirements.txt
├── .gitignore
└── README.md
```