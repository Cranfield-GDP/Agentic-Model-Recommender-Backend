from llm_provider import llm

def run(query: str) -> str:
    prompt = (
        "[Deployment Decision]\n"
        f"User requirements: {query}\n"
        "Advise whether to deploy on Cloud or MEC, explaining the reasoning (latency, scalability, cost)."
    )
    return llm(prompt)
