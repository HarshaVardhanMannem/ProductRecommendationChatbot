import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  # Ensure SerpAPI key is set

# Initialize LLM model
llm = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize vector store with embeddings
vector_store = Chroma(persist_directory="ecommerce_db", embedding_function=embeddings)

# Define a new prompt for conversation-based recommendations
conversation_prompt = """
You are an AI assistant helping customers with product recommendations.
Use the chat history to understand the context of follow-up questions.

Chat History:
{chat_history}

User's Latest Question:
{question}

Relevant Products:
{context}

Provide a response that considers the conversation history.
"""

# SerpAPI Web Search function
def web_search(query: str):
    search_url = f"https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google"  # You can use other engines as well
    }
    response = requests.get(search_url, params=params)
    results = response.json()
    
    # Extract relevant data from the SerpAPI response
    if "organic_results" in results:
        return [result["title"] + " - " + result["link"] for result in results["organic_results"][:5]]
    return []

# Define state for application
class State(TypedDict):
    question: str
    chat_history: str
    context: List[Document]  # Ensure 'context' is part of the state
    recommendations: str

def retrieve(state: State):
    # First, attempt to retrieve product info from vector store
    retrieved_docs = vector_store.similarity_search(state["question"], k=5)
    
    # If no results from vector store, search the web for additional information
    if not retrieved_docs:
        web_results = web_search(state["question"])
        # Return the web results as strings
        return {"context": [{"page_content": result} for result in web_results] if web_results else []}
    
    # If results from vector store exist, combine them with web search
    web_results = web_search(state["question"])  # Get web results
    combined_context = retrieved_docs + [{"page_content": result} for result in web_results]  # Combine docs with web results
    return {"context": combined_context}

def generate(state: State):
    docs_content = []
    
    # Loop through the context and ensure we handle both Document and dictionary types properly
    for doc in state.get("context", []):
        if isinstance(doc, Document):
            docs_content.append(doc.page_content)  # Extract page content from Document
        elif isinstance(doc, dict) and "page_content" in doc:
            docs_content.append(doc["page_content"])  # Extract page content from web search result (dictionary)
        else:
            docs_content.append(str(doc))  # Fallback to string conversion if necessary

    chat_context = state["chat_history"]
    
    prompt_text = conversation_prompt.format(
        chat_history=chat_context, 
        question=state["question"], 
        context="\n\n".join(docs_content) if docs_content else "No relevant products found."
    )

    response = llm.invoke(prompt_text)
    return {"recommendations": response.content}


# Compile application logic
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit Chatbot UI
def main():
    st.title("🛍️ E-commerce Product Recommender Chatbot")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input for query
    question = st.chat_input("What product are you looking for?")
    
    if question:
        chat_context = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history
        )

        # Store user message
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Ensure context is always passed
        response = graph.invoke({"question": question, "chat_history": chat_context, "context": []})
        answer = response["recommendations"]

        # Store chatbot response
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Refresh the UI
        st.rerun()

    # Terminate session button
    if st.button("End Chat Session"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()
