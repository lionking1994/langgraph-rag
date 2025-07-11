# test_simulate_clients.py

import concurrent.futures
from langgraph_agent import ask

# 10 diverse product-related and non-product questions
questions = [
    "What is the cheapest bread mix?",
    "How many gluten-free products are available?",
    "What is the most expensive product among them?",
    "Show me all cake mixes under $10.",
    "Which product has the highest rating?",
    "Tell me about the Apple Cinnamon Doughnut Mix.",
    "Are there any products out of stock?"
    "How much does the Lemon Bar Mix cost?",
    "What are the ingredients in the Pizza Crust Mix?",
    "Which products are on sale?"
]

def simulate_client(question):
    # Each client starts with an empty chat history
    answer, _ = ask(question, [])
    return f"Q: {question}\nA: {answer}\n{'-'*60}"

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(simulate_client, q) for q in questions]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()
