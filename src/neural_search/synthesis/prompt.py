def _get(chunk, key: str):
    """Access a chunk field whether it's a dict or an object."""
    if isinstance(chunk, dict):
        return chunk.get(key)
    return getattr(chunk, key, None)


def build_prompt(query: str, chunks: list) -> dict:
    if not query.strip():
        return {"system": "", "user": ""}

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source_file = _get(chunk, "source_file") or "unknown"
        page = _get(chunk, "page") or "?"
        text = _get(chunk, "text") or ""

        context_blocks.append(
            f"[Source {i}: {source_file}, Page {page}]\n{text}"
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
