def _get(chunk, key: str):
    """Access a chunk field whether it's a dict or a dataclass."""
    if isinstance(chunk, dict):
        return chunk[key]
    return getattr(chunk, key)


def build_prompt(query: str, chunks: list) -> dict:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Source {i}: {_get(chunk, 'source_file')}, "
            f"Page {_get(chunk, 'page')}]\n{_get(chunk, 'text')}"
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
