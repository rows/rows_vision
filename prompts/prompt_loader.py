# llm/prompt_loader.py
import os

def load_prompt(name: str) -> str:
    base_path = os.path.join(os.path.dirname(__file__),'')
    with open(os.path.join(base_path, f"{name}.txt"), "r") as file:
        return file.read()
