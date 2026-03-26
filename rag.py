"""
RAG retrieval: scoped knowledge base search with quality checks.

Flow:
1 Enrich the query for free with what we've learned
2. use the intent to search only the relevant chunks
3. Check to see how well we've retrieved information
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Retrieval quality thresholds
HIGH_CONFIDENCE = 0.75   # strong match, use this context
MEDIUM_CONFIDENCE = 0.50 # expand search to the full KB


def _embed(text: str) -> list[float]:
    """Embed a single string. Kept as a helper to avoid repetition."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return response.data[0].embedding

# enrich the user's query because it's free and improves retrieval
def _enrich_query(message: str, intent: str) -> str:
    if intent and intent != "unknown":
        return f"{intent} question: {message}"
    return message


def _search(collection, query_embedding: list[float], section: str | None, n: int) -> dict:
    """
    Run a Chroma similarity search, optionally scoped to a KB section.
    Returns the raw Chroma results dict.
    """
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n,
    }

    # Scoped search: only look in the relevant section
    # This is faster, cheaper, and more accurate than searching everything
    if section and section != "general":
        kwargs["where"] = {"section": section}

    return collection.query(**kwargs)


def retrieve(message: str, playbook: dict, collection) -> dict:
    """
    Main retrieval function. Called by the orchestrator with the message,
    the router's playbook, and the Chroma collection.

    Returns a dict with retrieved chunks and a quality assessment.
    """
    intent = playbook.get("intent", "unknown") #fallback to "unknown"
    section = playbook.get("kb_section", "general") #fallback to general 
    chunk_count = playbook.get("chunk_count", 5) #fallback to 5

    # Step 1: Enrich the query with intent context
    enriched_query = _enrich_query(message, intent)
    query_embedding = _embed(enriched_query)

    # Step 2: Scoped search — only the relevant KB section
    results = _search(collection, query_embedding, section, chunk_count)

    top_score = 1 - results["distances"][0][0]  # convert cosine distance to similarity
    chunks = results["documents"][0]
    scores = [1 - d for d in results["distances"][0]]

    # Step 3: Quality check
    if top_score >= HIGH_CONFIDENCE:
        return {
            "chunks": chunks,
            "scores": scores,
            "quality": "high",
            "search_scope": section,
            "query_used": enriched_query,
        }

    if top_score >= MEDIUM_CONFIDENCE:
        # Medium confidence — expand to full KB and retry
        expanded_results = _search(collection, query_embedding, section=None, n=chunk_count) # none removes the filter and searches the full KB
        expanded_score = 1 - expanded_results["distances"][0][0]
        expanded_chunks = expanded_results["documents"][0]
        expanded_scores = [1 - d for d in expanded_results["distances"][0]]

        return {
            "chunks": expanded_chunks,
            "scores": expanded_scores,
            "quality": "medium",
            "search_scope": "full_kb",    # flag that we had to expand
            "query_used": enriched_query,
        }

    # Low confidence — we found something but it's not relevant
    # Return what we have but flag it so the response layer can be cautious
    return {
        "chunks": chunks,
        "scores": scores,
        "quality": "low",
        "search_scope": section,
        "query_used": enriched_query,
    }


# Quick test
if __name__ == "__main__":
    from knowledge_base import KnowledgeBase

    kb = KnowledgeBase()

    test_cases = [
        {
            "message": "I was charged twice this month",
            "playbook": {"intent": "billing", "kb_section": "billing", "chunk_count": 3},
        },
        {
            "message": "My package hasn't arrived and it's been two weeks",
            "playbook": {"intent": "shipping", "kb_section": "shipping", "chunk_count": 3},
        },
        {
            "message": "The app keeps crashing when I upload a large file",
            "playbook": {"intent": "technical", "kb_section": "technical", "chunk_count": 3},
        },
        {
            "message": "What is the meaning of life?",   # should score low
            "playbook": {"intent": "unknown", "kb_section": "general", "chunk_count": 3},
        },
    ]

    print("=" * 60)
    print("RAG RETRIEVAL")
    print("=" * 60)

    for case in test_cases:
        result = retrieve(case["message"], case["playbook"], kb.collection)
        print(f"\nMessage: {case['message']}")
        print(f"  Quality:    {result['quality']}")
        print(f"  Scope:      {result['search_scope']}")
        print(f"  Top score:  {result['scores'][0]:.3f}")
        print(f"  Top chunk:  {result['chunks'][0][:80]}...")
