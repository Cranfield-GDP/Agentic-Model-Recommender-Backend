from llm_provider import llm

def run(query: str) -> str:
    prompt = (
        "[Requirement Analysis]\n"
        f"User input: {query}\n"
        "Extract telecom network requirements and ask clarifying questions if needed."
    )
    return llm(prompt)
