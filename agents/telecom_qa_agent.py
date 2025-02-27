from llm_provider import llm

def run(query: str) -> str:
    prompt = (
        "[Q&A]\n"
        f"User question: {query}\n"
        "Provide a clear and concise answer tailored to telecom requirements."
    )
    return llm(prompt)
