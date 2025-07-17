import os

# Central configuration for the project

class Config:
    # Project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Database
    DB_PATH = os.path.join(PROJECT_ROOT, "src", "db", "products.db")

    # LLM Model
    LLM_MODEL = "gpt-4o"
    LLM_TEMPERATURE = 0.1

    # Embedding Model
    EMBEDDING_MODEL = "text-embedding-3-small"

    # Vectorstore path (directory containing index.faiss and index.pkl)
    VECTORSTORE_PATH = os.path.join(PROJECT_ROOT, "src", "db", "faiss_mix")

    # Prompt directory
    PROMPTS_DIR = os.path.join(PROJECT_ROOT, "src", "prompts")

    # Dotenv path (absolute path to .env in project root)
    DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")

    # Add more config as needed 