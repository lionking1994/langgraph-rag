from typing import TypedDict, List, Dict, Any  
from langgraph.graph import StateGraph, END  
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
from langchain_community.vectorstores import FAISS  
from langchain.prompts import ChatPromptTemplate  
from query_processor import IntelligentQueryProcessor  
import json  
from dotenv import load_dotenv  
load_dotenv()  

# Initialize components  
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)  
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  
vectorstore = FAISS.load_local("faiss_mix", embeddings, allow_dangerous_deserialization=True)  
query_processor = IntelligentQueryProcessor()  

class AgentState(TypedDict):  
    query: str  
    query_type: str  
    needs_structured_data: bool  
    needs_semantic_search: bool  
    structured_results: str  
    semantic_results: str  
    final_answer: str  
    chat_history: List[Dict[str, str]]  
    last_product_query: str  
    last_product_answer: str  
    is_product_question: bool  
    is_product_followup: bool  
    last_node: str  # Track the last executed node  
    structured_complete: bool  # Track if structured query is done  
    semantic_complete: bool    # Track if semantic search is done  
    is_non_product: bool # New field for non-product classification

# Create graph  
graph = StateGraph(AgentState)  

def classify_and_route_node(state: AgentState) -> AgentState:  
    """
    Single LLM call to classify product, followup, and reasoning flags for routing.
    """
    last_product_query = state.get("last_product_query", "")
    last_product_answer = state.get("last_product_answer", "")
    query = state["query"]
    chat_context = ""
    if last_product_query:
        chat_context = f"Previous product question: {last_product_query}\nPrevious product answer: {last_product_answer[:200]}\n"
    prompt = ChatPromptTemplate.from_template("""
    {chat_context}
    Analyze the following query and context. Return a JSON object with these keys:
    - is_product_question: true if the query is about a product, item, price, cost, shopping, product info, or attributes (including recipes, ingredients, brands, categories, etc)
    - is_product_followup: true if the query refers to a previous product question/answer (e.g., uses reference words like 'it', 'this', 'that', or is a short follow-up)
    - needs_structured_data: true if the query asks for counts, prices, comparisons, filters, or specific attributes
    - needs_semantic_search: true if the query asks for recommendations, descriptions, how-to, general info, or explanations
    - query_type: main type (counting/pricing/filtering/searching/comparing/explaining)
    - is_non_product: true if the query is NOT about products, shopping, or product info (e.g., greetings, general questions, etc)
    
    Query: {query}
    
    Return JSON: {{
      "is_product_question": bool,
      "is_product_followup": bool,
      "needs_structured_data": bool,
      "needs_semantic_search": bool,
      "query_type": "type",
      "is_non_product": bool
    }}
    """)
    response = llm.invoke(prompt.format(query=query, chat_context=chat_context)).content
    import re
    response_clean = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()  
    try:  
        data = json.loads(response_clean)  
        state["is_product_question"] = data.get("is_product_question", False)
        state["is_product_followup"] = data.get("is_product_followup", False)
        state["needs_structured_data"] = data.get("needs_structured_data", True)  
        state["needs_semantic_search"] = data.get("needs_semantic_search", False)  
        state["query_type"] = data.get("query_type", "general")  
        state["is_non_product"] = data.get("is_non_product", False)
    except Exception as e:  
        print("classify_and_route_node JSON decode error:", e)  
        state["is_product_question"] = False
        state["is_product_followup"] = False
        state["needs_structured_data"] = True  
        state["needs_semantic_search"] = True  
        state["query_type"] = "general"  
        state["is_non_product"] = True
    print("classify_and_route_node state after classification:", state)  
    state["last_node"] = "classify_and_route"  
    return state  

def reasoning_router(state: AgentState) -> str:  
    if state.get("is_non_product", False):
        return "non_product"
    needs_structured = state.get("needs_structured_data", False)  
    needs_semantic = state.get("needs_semantic_search", False)  
    has_structured = state.get("structured_complete", False)  
    has_semantic = state.get("semantic_complete", False)  
    if needs_structured and not has_structured:  
        return "structured_query"  
    if needs_semantic and not has_semantic:  
        return "semantic_search"  
    if (not needs_structured or has_structured) and (not needs_semantic or has_semantic):  
        return "synthesize"  
    return "synthesize"  

def structured_query(state: AgentState) -> AgentState:
    if state.get("needs_structured_data", False):
        context = {}
        query_to_use = state["query"]
        # If this is a follow-up, inject last product context into the query
        if state.get("is_product_followup") and state.get("last_product_query"):
            context = {
                "last_product_query": state.get("last_product_query", ""),
                "last_product_answer": state.get("last_product_answer", "")
            }
            # If the query is short or contains reference words, append the last product
            reference_words = ["it", "this", "that", "them", "these", "those"]
            if any(word in query_to_use.lower() for word in reference_words) or len(query_to_use.split()) <= 5:
                query_to_use = f"{state['last_product_query']} {query_to_use}"
        result = query_processor.process_query(query_to_use, context)
        state["structured_results"] = result
    state["structured_complete"] = True
    state["last_node"] = "structured_query"
    return state

def semantic_search(state: AgentState) -> AgentState:
    if state.get("needs_semantic_search", False):
        search_query = state["query"]
        if state.get("is_product_followup") and state.get("last_product_query"):
            reference_words = ["it", "this", "that", "them", "these", "those"]
            if any(word in search_query.lower() for word in reference_words) or len(search_query.split()) <= 5:
                search_query = f"{state['last_product_query']} {search_query}"
        docs = vectorstore.similarity_search(search_query, k=5)
        context = "\n---\n".join([
            f"Product: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        ])
        state["semantic_results"] = context
    state["semantic_complete"] = True
    state["last_node"] = "semantic_search"
    return state

def format_final_answer(answer: str, user_query: str = "") -> str:  
    import re  
    count_patterns = [  
        r'^how many', r'number of', r'count of', r'how much', r'what is the total', r'total number', r'how often', r'how frequently'  
    ]  
    is_count_query = any(re.search(pat, user_query.strip().lower()) for pat in count_patterns)  
    if is_count_query:  
        format_prompt = ChatPromptTemplate.from_template("""
        The user asked: {user_query}  
        Here is the raw answer from the database or context:  
        {raw_answer}  
        Please return ONLY single sentence, with no list or extra details. Do not include any product list, markdown, or explanations. Do not include disclaimers, hedging, or meta-comments. Only show the direct answer to the user's question.  
        """)  
    else:  
        format_prompt = ChatPromptTemplate.from_template("""
        The user asked: {user_query}  
        Here is the raw answer from the database or context:  
        {raw_answer}  
        Please reformat the answer for clear, readable viewing in a chat UI. Use Markdown where appropriate:  
        - If the answer contains a list of products, you MUST display it as a formatted list. Do not summarize it or omit it.
        - Format product lists as a Markdown bullet list: '- Product Name: **$Price**'.  
        - Bold prices (e.g., **$55**), but do not bold product names.  
        - Use paragraphs and line breaks for clarity.  
        - Do not include any disclaimers, hedging, or meta-comments.  
        """)  
    formatted = llm.invoke(format_prompt.format(raw_answer=answer, user_query=user_query)).content  
    disclaimer_keywords = [  
        r'discrepancy', r'context', r'verify', r'database results', r'If you have access',  
        r'additional context', r'provided', r'helpful to', r'uncertain', r'not sure', r'cannot verify',  
        r'If you have any more questions', r'If you need', r'If you require', r'If you would like',  
        r'If you have access', r'If you have further', r'If you want', r'If you need more',  
        r'If you have access to the complete database', r'If you have access to more information',  
        r'If you have access to further details', r'If you have access to the full database',  
        r'If you have access to the full context', r'If you have access to the full data',  
        r'If you have access to the full product list', r'If you have access to the full information',  
        r'If you have access to the full details', r'If you have access to the full records',  
        r'If you have access to the full set', r'If you have access to the full source',  
    ]  
    pattern = re.compile(r'[^.]*(' + '|'.join(disclaimer_keywords) + r')[^.]*[\.!?]', re.IGNORECASE)  
    formatted = pattern.sub('', formatted)  
    formatted = formatted.replace('∗', '*')  
    formatted = re.sub(r'\n{3,}', '\n\n', formatted)  
    formatted = re.sub(r'\n\s*\n', '\n\n', formatted)  
    return formatted.strip()  

def synthesize_answer(state: AgentState) -> AgentState:  
    context_info = ""  
    if state.get("is_product_followup") and state.get("last_product_query"):  
        context_info = f"\nPrevious question: {state['last_product_query']}\nPrevious answer: {state['last_product_answer']}\n"  
    if state.get("structured_results") and not state.get("semantic_results"):  
        if state.get("is_product_followup"):  
            prompt = ChatPromptTemplate.from_template("""
            {context_info}  
            Current question: {query}  
            Current results: {structured}  
            Provide a helpful answer that summarizes the facts. Do not include disclaimers, hedging, or comments about possible discrepancies. If there is no data, simply state so directly.
            """)  
            answer = llm.invoke(prompt.format(  
                context_info=context_info,  
                query=state["query"],  
                structured=state["structured_results"]  
            )).content  
            state["final_answer"] = format_final_answer(answer, state["query"])  
        else:  
            state["final_answer"] = format_final_answer(state["structured_results"], state["query"])  
    elif state.get("semantic_results") and not state.get("structured_results"):  
        prompt = ChatPromptTemplate.from_template("""
        {context_info}  
        Question: {query}  
        Based on this information:  
        {context}  
        Provide a helpful answer that summarizes the facts. Do not include disclaimers, hedging, or comments about possible discrepancies. If there is no data, simply state so directly.
        """)  
        answer = llm.invoke(prompt.format(  
            context_info=context_info,  
            query=state["query"],  
            context=state["semantic_results"]  
        )).content  
        state["final_answer"] = format_final_answer(answer, state["query"])  
    elif state.get("structured_results") and state.get("semantic_results"):  
        prompt = ChatPromptTemplate.from_template("""
        {context_info}  
        Question: {query}  
        Database Results:  
        {structured}  
        Additional Context:  
        {semantic}  
        Combine both sources to provide a comprehensive, direct answer that only summarizes the facts. Do not include disclaimers, hedging, or comments about possible discrepancies. If there is no data, simply state so directly.
        """)  
        answer = llm.invoke(prompt.format(  
            context_info=context_info,  
            query=state["query"],  
            structured=state["structured_results"],  
            semantic=state["semantic_results"]  
        )).content  
        state["final_answer"] = format_final_answer(answer, state["query"])  
    else:  
        state["final_answer"] = "I'm sorry, I couldn't find relevant information to answer your question."  
    print("synthesize_answer final_answer:", state["final_answer"])
    state["last_node"] = "synthesize"  
    return state  

def handle_non_product(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant. Answer the following user question conversationally and helpfully. Do NOT mention products, shopping, or bakery unless the user asks about them.
    
    User question: {query}
    """)
    answer = llm.invoke(prompt.format(query=state["query"])).content
    state["final_answer"] = answer
    state["last_node"] = "non_product"
    return state

# Build the graph  
graph.add_node("classify_and_route", classify_and_route_node)  
graph.add_node("structured_query", structured_query)  
graph.add_node("semantic_search", semantic_search)  
graph.add_node("synthesize", synthesize_answer)  
graph.add_node("non_product", handle_non_product)  

graph.set_entry_point("classify_and_route")

# Only use reasoning_router for routing from classify_and_route
# Remove any reference to 'reasoning' node

graph.add_conditional_edges(
    "classify_and_route",
    reasoning_router,
    {
        "structured_query": "structured_query",
        "semantic_search": "semantic_search",
        "synthesize": "synthesize",
        "non_product": "non_product"
    }
)

graph.add_edge("structured_query", "classify_and_route")
graph.add_edge("semantic_search", "classify_and_route")
graph.add_edge("synthesize", END)
graph.add_edge("non_product", END)

app = graph.compile()  

def visualize_graph():  
    ascii_diagram = app.get_graph().draw_ascii()  
    mermaid_code = app.get_graph().draw_mermaid()  
    return ascii_diagram, mermaid_code  

def get_graph_mermaid_png():  
    try:  
        return app.get_graph().draw_mermaid_png()  
    except Exception:  
        return None  

def get_last_product_context(chat_history: List[Dict[str, str]]) -> tuple[str, str]:
    last_query = ""
    last_answer = ""
    for i in range(len(chat_history) - 1, -1, -1):
        msg = chat_history[i]
        if msg.get("role") == "assistant":
            content = msg.get("content", "").lower()
            msg_type = msg.get("type", "").lower()
            is_product_response = (
                msg_type == "product" or 
                any(keyword in content for keyword in ["product", "price", "$", "cost", "expensive", "cheap", "item"])
            )
            if is_product_response:
                last_answer = msg.get("content", "")
                for j in range(i - 1, -1, -1):
                    if chat_history[j].get("role") == "user":
                        last_query = chat_history[j].get("content", "")
                        break
                break
    print(f"Found last product context - Query: '{last_query}', Answer: '{last_answer[:50]}...'")
    return last_query, last_answer

def ask(query: str, chat_history: List[Dict[str, str]] = None) -> tuple[str, List[Dict[str, str]]]:
    if chat_history is None:
        chat_history = []
    last_product_query, last_product_answer = get_last_product_context(chat_history)
    initial_state = {
        "query": query,
        "query_type": "",
        "needs_structured_data": False,
        "needs_semantic_search": False,
        "structured_results": "",
        "semantic_results": "",
        "final_answer": "",
        "chat_history": chat_history,
        "last_product_query": last_product_query,
        "last_product_answer": last_product_answer,
        "is_product_question": False,
        "is_product_followup": False,
        "is_non_product": False,
        "last_node": "START",
        "structured_complete": False,
        "semantic_complete": False
    }
    result = app.invoke(initial_state)
    chat_history.append({"role": "user", "content": query})
    message_type = "product" if (result.get("is_product_question", False) or result.get("is_product_followup", False)) else "non-product"
    if result.get("is_non_product", False):
        message_type = "non-product"
    chat_history.append({
        "role": "assistant",
        "content": result.get("final_answer", "Sorry, I couldn't process your question."),
        "type": message_type
    })
    return result.get("final_answer", "Sorry, I couldn't process your question."), chat_history        