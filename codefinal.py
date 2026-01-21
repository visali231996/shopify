import operator
import os
from typing import Dict, List, TypedDict, Annotated, Literal

from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain.messages import ToolMessage
from langgraph.graph import StateGraph, START, END

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ===========================
# ENVIRONMENT SETUP
# ===========================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "shopify_products"

qdrant = QdrantClient(url=QDRANT_URL)

last_products_cache: Dict[str, Dict] = {}

CATEGORY_KEYWORDS = {
    "mobile": ["mobile", "phone", "smartphone", "android", "iphone"],
    "laptop": ["laptop", "notebook", "macbook", "ultrabook"],
    "tablet": ["tablet", "ipad"],
    "accessory": ["charger", "cable", "earbuds", "headphones"]
}

def detect_category(keyword: str | None) -> str | None:
    if not keyword:
        return None

    keyword = keyword.lower()

    for category, words in CATEGORY_KEYWORDS.items():
        if any(w in keyword for w in words):
            return category

    return None

# ===========================
# TOOL 1 — FILTER PRODUCTS
# ===========================

from difflib import get_close_matches

@tool
def filter_products(keyword: str = None, price_min: float = 0.0, price_max: float = 999999.0):
    """
    Fetch products from Qdrant and filter by intent (derived from tags/title) and price.
    """

    records, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=300,
        with_payload=True
    )

    keyword = (keyword or "").lower()
    intent_category = detect_category(keyword)

    filtered = []

    for r in records:
        payload = r.payload or {}

        name = payload.get("title", "")
        handle = payload.get("handle")
        vendor = payload.get("vendor", "Unknown")
        price = payload.get("price", 0)

        tags = payload.get("tags", [])
        tags_lower = " ".join(tags).lower()
        title_lower = name.lower()

        if not handle:
            continue

        try:
            price = float(price)
        except:
            continue

        # ✅ CATEGORY MATCH (USING TAGS + TITLE ONLY)
        if intent_category:
            category_keywords = CATEGORY_KEYWORDS[intent_category]
            if not any(
                kw in tags_lower or kw in title_lower
                for kw in category_keywords
            ):
                continue

        # ✅ PRICE FILTER
        if not (price_min <= price <= price_max):
            continue

        filtered.append({
            "Product Name": name,
            "Vendor": vendor,
            "Handle": handle,
            "Price": price,
            "Tags": tags
        })

    # ✅ cache only what was actually shown
    st.session_state.last_products_cache = {
        p["Handle"]: p for p in filtered
}

    return filtered[:10]



# ===========================
# TOOL 2 — INVENTORY CHECK
# ===========================
import re

@tool
def check_inventory(product: str) -> Dict:
    """
    Checks availability of a product using cache first, then Qdrant.
    """

    normalized = product.strip().lower()

    # 1️⃣ Check cache first (shown products)
    cache = st.session_state.get("last_products_cache", {})
    if normalized in cache:
        return {
            "available": True,
            "message": "✅ Product is available",
            "handle": normalized,
            "source": "cache"
        }

    # 2️⃣ Fallback to Qdrant (GLOBAL lookup)
    results, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=200,
        with_payload=True
    )

    for r in results:
        payload = r.payload or {}
        handle = payload.get("handle", "").lower()
        title = payload.get("title", "").lower()

        if normalized == handle or normalized == title:
            return {
                "available": True,
                "message": "✅ Product is available",
                "handle": handle,
                "source": "qdrant"
            }

    return {
        "available": False,
        "message": "❌ Product not found in inventory."
    }


# ===========================
# TOOL 3 — CHECKOUT
# ===========================
from langchain.tools import tool
@tool
def checkout(handle: str) -> Dict:
    """
    helps the product add to the cart
    """
    cache = st.session_state.get("last_products_cache", {})

    if handle not in cache:
        return {
            "success": False,
            "message": "❌ Please select a product from the recommended list"
        }

    p = cache[handle]

    return {
        "success": True,
        "action": "add_to_cart",
        "Product Name": p["Product Name"],
        "Handle": p["Handle"],
        "Price": p["Price"]
    }


# ===========================
# TOOL LIST
# ===========================
tools = [filter_products, check_inventory, checkout]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# ===========================
# LANGGRAPH STATE DEFINITION
# ===========================
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

available_handles = list(last_products_cache.keys())
available_names = [p["Product Name"] for p in last_products_cache.values()]
# ===========================
# NODE 1 — LLM ReasonING NODE
# ===========================
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="""You are a sales assistant.
                                   

                                    STRICT RULES:
                                    RULES (NON-NEGOTIABLE):
                                    - If user intent is buying or browsing → CALL filter_products
                                    -- If user mentions ANY available handle → CALL check_inventory
                                    - NEVER ask questions before calling filter_products
                                    - NEVER respond in text when products are requested
                                    - Inventory & checkout ONLY work after products are shown
                                    - If no products were shown, reply:"Please ask to see products first."
                                    - Products MUST come from Qdrant only.
                                    -If the user wants to buy, browse, or see products,you MUST call the tool `filter_products`
                                    - Pass the user request as the keyword argument
                                    - NEVER invent product names or handles.
                                    - ONLY display products returned by filter_products in tabular form
                                    - ALWAYS display Product Name + Handle +tags+price together
                                    

                                    Category mapping:
                                    - mobile, phone, smartphone → mobile
                                    - laptop, notebook → laptop

                         """
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# ===========================
# NODE 2 — TOOL EXECUTION
# ===========================


###########
import json

def tool_node(state: dict):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])

        result.append(
            ToolMessage(
                content=json.dumps(observation),  # ✅ JSON, not str()
                tool_call_id=tool_call["id"]
            )
        )
    return {"messages": result}


# ===========================
# CONTROL FLOW
# ===========================
def should_continue(state):
    last = state["messages"][-1]

    # ✅ Only go to tool_node if a tool was actually requested
    if last.tool_calls:
        return "tool_node"

    return END



# ===========================
# MEMORY (OPTION A)
# ===========================
conversation_memory: list[AnyMessage] = []  # stores all messages in memory

# ===========================
# MAIN RESPONSE FUNCTION
# ===========================
def get_response(user_input: str) -> str:
    global conversation_memory

    # Add user input to memory
    conversation_memory.append(HumanMessage(content=user_input))

    # Initialize with full conversation history
    state: MessagesState = {
        "messages": conversation_memory,
        "llm_calls": 0
    }

    # Build the LangGraph agent
    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent_builder.add_edge("tool_node", "llm_call")
    agent = agent_builder.compile()

    # Run the graph
    result_state = agent.invoke(state)

    # Get the model's reply
    final_messages = result_state.get("messages", [])
    if final_messages:
        last_msg = final_messages[-1]

        # Save LLM reply to memory
        conversation_memory.append(last_msg)

        if isinstance(last_msg, ToolMessage):
            return last_msg.content  # raw dict as string

        return last_msg.content
    return "Sorry, I couldn't process your request."

# ===========================
# TESTING / DEMO
# ===========================

import streamlit as st

response = get_response("I want to buy laptop ")
print("Agent:", response)
# ===========================
# STREAMLIT APP
# ===========================
import streamlit as st
import pandas as pd

# ===========================
# STREAMLIT CONFIG
# ===========================
st.set_page_config(
    page_title="🛍️ AI Sales Assistant",
    page_icon="🤖",
    layout="wide"
)
with st.expander("🧪 Debug: Qdrant Collection"):
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        with_payload=True,
        with_vectors=False
    )

    debug_rows = []
    for p in points:
        payload = p.payload
        debug_rows.append({
            "ID": p.id,
            "Title": payload.get("title"),
            "Price": payload.get("price"),
            "Handle": payload.get("handle"),
            "Tags": payload.get("tags")
        })

    st.dataframe(debug_rows, use_container_width=True)

# ===========================
# SESSION STATE INIT
# ===========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_products_cache" not in st.session_state:
    st.session_state.last_products_cache = {}

if "cart" not in st.session_state:
    st.session_state.cart = []

# ===========================
# LAYOUT
# ===========================
col1, col2 = st.columns([2, 1], gap="large")

# ===========================
# LEFT — CHAT UI
# ===========================
with col1:
    st.title("🤖 AI Sales Assistant")
    st.caption("Powered by LangGraph + Qdrant")

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ===========================
# CHAT INPUT (ONLY ONCE, OUTSIDE LOOPS)
# ===========================
user_input = st.chat_input(
    "What are you looking for today?",
    key="main_chat_input"
)


if user_input:
    # Save user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    # Generate assistant response
    with st.spinner("Thinking..."):
        response = get_response(user_input)

        import json
        try:
            parsed = json.loads(response)
            if parsed.get("action") == "add_to_cart" and parsed.get("success"):
                st.session_state.cart.append({
                "Product Name": parsed["Product Name"],
                "Handle": parsed["Handle"],
                "Price": parsed["Price"],
    })

                st.session_state.cart.append({
                    "Product Name": parsed["Product Name"],
                    "Handle": parsed["Handle"],
                    "Price": parsed["Price"],
                    "Qdrant ID": parsed["point_id"]
                })
            else:
                st.warning("Product not verified from Qdrant")

        except Exception as e:
            print("Cart parse error:", e)


    # Save assistant response
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
    })

    # Force clean UI refresh
    st.rerun()

# ===========================
# RIGHT — CART
# ===========================





