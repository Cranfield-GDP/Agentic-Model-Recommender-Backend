from llm_provider import llm

def run() -> str:
    prompt = (
        "[Follow-Up]\n"
        "Summarize the conversation so far and ask the user if they have any further questions or need additional clarification."
    )
    return llm(prompt)
