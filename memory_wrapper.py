from langchain_core.memory import BaseMemory
from pydantic import PrivateAttr

class MemorySaverWrapper(BaseMemory):
    memory_key: str = "chat_history"
    _memory_saver: object = PrivateAttr()
    _history: str = PrivateAttr(default="")

    def __init__(self, memory_saver, **kwargs):
        super().__init__(**kwargs)
        self._memory_saver = memory_saver
        self._history = ""  # Initialize conversation history

    def load_memory_variables(self, inputs: dict) -> dict:
        """Return the conversation history as a dict."""
        return {self.memory_key: self._history}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Append new conversation turns to the history."""
        new_message = f"Human: {inputs.get('input', '')}\nAI: {outputs.get('output', '')}\n"
        self._history += new_message

    @property
    def memory_variables(self) -> list:
        """Return the list of keys that this memory object provides."""
        return [self.memory_key]

    def clear(self) -> None:
        """Clear the conversation history."""
        self._history = ""

    @property
    def buffer(self):
        """Return the conversation history as a string."""
        return self._history
