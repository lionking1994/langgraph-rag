import os
from src.config import Config

class PromptService:
    def __init__(self, prompts_dir=None):
        # Use env var if set, else argument, else config
        prompts_dir = os.environ.get('PROMPTS_DIR', prompts_dir or Config.PROMPTS_DIR)
        self.prompts_dir = prompts_dir

    def get_prompt(self, name: str) -> str:
        path = os.path.join(self.prompts_dir, name)
        with open(path, 'r', encoding='utf-8') as f:
            return f.read() 