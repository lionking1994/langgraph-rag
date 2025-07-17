from . import llm, load_prompt
import re

def response_synthesis_node(state):
    context_info = ""
    if state.get("is_product_followup") and state.get("last_product_query"):
        # Extract the specific product name from the previous answer if possible
        last_answer = state['last_product_answer']
        specific_product = ""
        
        # Try to extract product name from the previous answer
        if "**" in last_answer:
            # Look for bold product names (format: **Product Name**: **$Price**)
            product_matches = re.findall(r'\*\*([^*]+)\*\*: \*\*\$', last_answer)
            if product_matches:
                specific_product = product_matches[0].strip()
        
        context_info = f"\nPrevious question: {state['last_product_query']}\nPrevious answer: {state['last_product_answer']}\n"
        if specific_product:
            context_info += f"Specific product being referenced: {specific_product}\n"
    if state.get("structured_results") and not state.get("semantic_results"):
        # Do not call LLM here; just pass the raw SQL result to the formatter
        state["final_answer"] = state["structured_results"]
    elif state.get("semantic_results") and not state.get("structured_results"):
        base_answer = state["semantic_results"]
        prompt_template = load_prompt("semantic_search.txt")
        from langchain.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(prompt_template)
        answer = llm.invoke(prompt.format(
            context_info=context_info,
            user_query=state["query"],
            raw_answer=base_answer
        )).content
        state["final_answer"] = answer.strip()
    elif state.get("structured_results") and state.get("semantic_results"):
        prompt_template = load_prompt("synthesize_answer.txt")
        from langchain.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(prompt_template)
        answer = llm.invoke(prompt.format(
            context_info=context_info,
            user_query=state["query"],
            structured=state["structured_results"],
            semantic=state["semantic_results"]
        )).content
        state["final_answer"] = answer.strip()
    else:
        state["final_answer"] = "I'm sorry, I couldn't find relevant information to answer your question."
    state["last_node"] = "response_synthesis"
    return state 