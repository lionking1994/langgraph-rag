import os
from dotenv import load_dotenv
from src.config import Config
load_dotenv(dotenv_path=Config.DOTENV_PATH)

from src.models.agent_state import AgentState
from src.tools.query_processor import IntelligentQueryProcessor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
import json
import re

llm = ChatOpenAI(model=os.environ.get('LLM_MODEL', Config.LLM_MODEL), temperature=float(os.environ.get('LLM_TEMPERATURE', Config.LLM_TEMPERATURE)))
embeddings = OpenAIEmbeddings(model=os.environ.get('EMBEDDING_MODEL', Config.EMBEDDING_MODEL))
vectorstore = FAISS.load_local(os.environ.get('VECTORSTORE_PATH', Config.VECTORSTORE_PATH), embeddings, allow_dangerous_deserialization=True)
query_processor = IntelligentQueryProcessor()

def load_prompt(filename):
    import os
    with open(os.path.join(os.path.dirname(__file__), '../prompts', filename), 'r', encoding='utf-8') as f:
        return f.read()

from .reasoning_node import reasoning_node
from .reasoning_router import reasoning_router
from .data_retrieval_node import data_retrieval_node
from .knowledge_search_node import knowledge_search_node
from .response_synthesis_node import response_synthesis_node
from .general_chat_node import general_chat_node

__all__ = [
    'reasoning_node',
    'reasoning_router',
    'data_retrieval_node',
    'knowledge_search_node',
    'response_synthesis_node',
    'general_chat_node',
    'llm',
    'embeddings',
    'vectorstore',
    'query_processor',
    'load_prompt',
] 