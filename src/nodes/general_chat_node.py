from . import llm, load_prompt

def general_chat_node(state):
    prompt_template = load_prompt("non_product.txt")
    from langchain.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(prompt_template)
    answer = llm.invoke(prompt.format(query=state["query"])).content
    state["final_answer"] = answer
    state["last_node"] = "general_chat"
    return state 