def build_prompt(query: str, chunks: list[dict]) -> dict:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Source {i}: {chunk['source_file']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    system = (
        "You are a precise document assistant. "
        "Answer the user's question using ONLY the provided context. "
        "Always cite which source and page your answer comes from. "
        "If the answer is not present in the context, respond with: "
        "'I could not find this information in the provided documents.'"
    )

    user = f"Question: {query}\n\nContext:\n{context}"

    return {"system": system, "user": user}
