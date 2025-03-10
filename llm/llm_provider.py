from llm.config import Config

from llm.llm_model import LLMModel

_config = None

def set_config(config: Config):
    global _config
    _config = config

def execute_llm(prompt: str):
    global _config
    
    if _config is None:
        raise ValueError("LLM config not set. Please call set_config() first.")
    
    llmModel = LLMModel(_config.model)

    return llmModel.llm.invoke(prompt)

def llm(prompt: str):
    llm_output = execute_llm(prompt=prompt)
    return llm_output