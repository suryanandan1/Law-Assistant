from loguru import logger


class ContextBuilder:
    def __init__(self, max_context_chars=12000):
        self.max_context_chars = max_context_chars

    def remove_duplicates(self, chunks):
        seen = set()
        return [chunk for chunk in chunks if not (text := chunk["text"].strip()) in seen and not seen.add(text)]

    def build_context(self, question, retrieved_chunks):
        chunks = self.remove_duplicates(sorted(retrieved_chunks, key=lambda chunk: chunk["score"], reverse=True))
        context_parts, pages, current_size = [], set(), 0
        for chunk in chunks:
            chunk_text = f"[PAGE {chunk['page']}]\n{chunk['text']}\n"
            if current_size + len(chunk_text) > self.max_context_chars:
                break
            context_parts.append(chunk_text)
            pages.add(chunk["page"])
            current_size += len(chunk_text)

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a legal document assistant.

STRICT RULES:

1. Answer ONLY from the supplied context.
2. Do NOT use outside knowledge.
3. Do NOT invent information.
4. If the answer is missing, say:

"The document does not contain this information."

5. Always cite page numbers.
6. Explain supporting evidence.

QUESTION:

{question}

CONTEXT:

{context}

OUTPUT FORMAT:

Answer:
<answer>

Supporting Pages:
<pages>

Confidence:
<High/Medium/Low>
"""

        logger.info(f"Context Size: {len(context)} chars")
        return {"prompt": prompt, "pages": sorted(pages)}
