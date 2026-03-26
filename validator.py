"""
Response validation: guardrails and grounding checks.
Runs after response generation, before sending to the customer.

Three layers (cheapest first):
1. Rule-based checks — forbidden phrases, character breaks, strong commitments (free)
2. Hallucination check — numbers/amounts in response not found in KB chunks (free, uses regex)
3. Grounding check — embedding similarity between response and retrieved context (one API call)

Returns an action:
  "send"               → response passed, safe to send
  "retry_bigger_model" → issues found, worth retrying with a more capable model
  "escalate_human"     → already on best model or critical failure, route to human

The validator only judges — it does not retry or escalate itself.
The orchestrator acts on the returned action.
"""

import re
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Grounding threshold — response embedding must be at least this similar
# to the retrieved chunks to be considered grounded in the KB
GROUNDING_THRESHOLD = 0.60

# Forbidden patterns — things that should never appear in a customer response
FORBIDDEN_PATTERNS = [
    (r"as an ai", "Model broke character (mentioned being an AI)"),
    (r"as a language model", "Model broke character"),
    (r"i (cannot|can't) (help|assist) with that", "Unhelpful refusal without escalation"),
    (r"i (guarantee|promise) ", "Overly strong commitment — legal risk"),
    (r"100% (certain|sure|guaranteed)", "Overly strong commitment — legal risk"),
    (r"(competitor_a|competitor_b|rival_brand)", "Competitor mention"),  # swap in real names
]

# Regex patterns for extracting specific claims from the response
# Used to cross-check against KB chunks
CLAIM_PATTERNS = {
    "dollar_amount": r'\$\d+\.?\d*',
    "date": r'\b\d{4}-\d{2}-\d{2}\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b',
    "days": r'\b\d+\s+(business\s+)?days?\b',
    "percentage": r'\b\d+%',
}


def _check_rules(response_text: str) -> list[dict]:
    """
    Rule-based checks against forbidden patterns.
    Fast and free — runs first before any embedding calls.
    """
    issues = []
    response_lower = response_text.lower()

    for pattern, reason in FORBIDDEN_PATTERNS:
        if re.search(pattern, response_lower):
            issues.append({
                "type": "forbidden_phrase",
                "detail": reason,
                "severity": "high",
            })

    return issues


def _check_hallucinations(response_text: str, chunks: list[str]) -> list[dict]:
    """
    Extract specific claims (dollar amounts, dates, timeframes) from the response
    and check whether they appear in the retrieved KB chunks.

    If a specific claim is in the response but not in any chunk, it may be fabricated.
    This is not foolproof — the model may correctly know something not in the chunks —
    but it's a cheap signal worth flagging.
    """
    issues = []
    chunks_combined = " ".join(chunks).lower()

    for claim_type, pattern in CLAIM_PATTERNS.items():
        matches = re.findall(pattern, response_text.lower())
        for match in matches:
            # Flatten tuple matches (from groups) to strings
            match_str = match if isinstance(match, str) else " ".join(match).strip()
            if match_str and match_str not in chunks_combined:
                issues.append({
                    "type": "potential_hallucination",
                    "detail": f"{claim_type} '{match_str}' in response not found in KB chunks",
                    "severity": "medium",
                })

    return issues


def _check_grounding(response_text: str, chunks: list[str]) -> float:
    """
    Embed the response and compare to each retrieved chunk.
    Returns the highest similarity score across all chunks.

    High score = response is well-grounded in the KB
    Low score = response may have drifted from the source material
    """
    if not chunks:
        return 0.0

    # Embed response and all chunks in one batched call
    texts = [response_text] + chunks
    result = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )

    embeddings = [item.embedding for item in result.data]
    response_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]

    # Cosine similarity (dot product of normalized vectors)
    def cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    scores = [cosine_similarity(response_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
    return max(scores)


def validate(
    response_text: str,
    retrieval: dict,
    generation: dict,
) -> dict:
    """
    Run all validation checks and return a recommended action.

    response_text: the generated response string
    retrieval:     the dict returned by rag.retrieve()
    generation:    the dict returned by response.generate()
    """
    chunks = retrieval.get("chunks", [])
    retrieval_quality = retrieval.get("quality", "high")
    current_model = generation.get("model", "gpt-4o-mini")
    tools_were_called = len(generation.get("tools_called", [])) > 0

    issues = []

    # Layer 1: Rule-based (free)
    issues.extend(_check_rules(response_text))

    # Layer 2: Hallucination check (free)
    # Skip if tools were called — specific values in the response (dates, amounts)
    # came from tool results (ground truth from internal systems), not the model's
    # imagination. Flagging those as hallucinations would be a false positive.
    # Also skip if retrieval quality was low (same reasoning applies).
    if not tools_were_called and retrieval_quality != "low":
        issues.extend(_check_hallucinations(response_text, chunks))

    # Layer 3: Grounding check (one embedding call)
    # Skip for short responses — clarifying questions asking for an order ID or email
    # are not making claims from the KB. Applying grounding to them would flag every
    # "Could you share your order ID?" as ungrounded, which is a false positive.
    # Also skip when tools were called — the response is grounded in tool results
    # (real system data), not KB chunks. Same reasoning as the hallucination skip above.
    grounding_score = _check_grounding(response_text, chunks)
    if not tools_were_called and retrieval_quality != "low" and len(response_text) > 200 and grounding_score < GROUNDING_THRESHOLD:
        issues.append({
            "type": "grounding",
            "detail": f"Response not well-grounded in KB (similarity: {grounding_score:.3f})",
            "severity": "medium",
        })

    # No issues — safe to send
    if not issues:
        return {
            "passed": True,
            "action": "send",
            "issues": [],
            "grounding_score": round(grounding_score, 3),
        }

    # Issues found — determine action based on severity and current model
    high_severity = any(i["severity"] == "high" for i in issues)

    if high_severity or current_model == "gpt-4o":
        # High severity or already on best model → human escalation
        action = "escalate_human"
    else:
        # On mini with medium issues → retry with bigger model
        action = "retry_bigger_model"

    return {
        "passed": False,
        "action": action,
        "issues": issues,
        "grounding_score": round(grounding_score, 3),
    }


# Quick test
if __name__ == "__main__":
    # Simulate a good response
    good_response = "I can see there was a duplicate charge on March 1st. I'll process a refund to your original payment method within 5-7 business days."
    good_chunks = [
        "If you were charged twice, contact support with your order ID and we will reverse the duplicate charge immediately.",
        "Refunds are processed within 5-7 business days and returned to the original payment method.",
    ]

    # Simulate a bad response (hallucinated amount, forbidden phrase)
    bad_response = "As an AI, I can tell you that you'll receive a $150.00 refund within 2 days. I guarantee this will be resolved."
    bad_chunks = good_chunks

    print("=" * 60)
    print("RESPONSE VALIDATION")
    print("=" * 60)

    for label, response_text, chunks in [
        ("GOOD RESPONSE", good_response, good_chunks),
        ("BAD RESPONSE", bad_response, bad_chunks),
    ]:
        result = validate(
            response_text,
            {"chunks": chunks, "quality": "high"},
            {"model": "gpt-4o-mini"},
        )
        print(f"\n{label}: {response_text[:80]}...")
        print(f"  Passed:   {result['passed']}")
        print(f"  Action:   {result['action']}")
        print(f"  Grounding: {result['grounding_score']}")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"  Issue: [{issue['severity']}] {issue['detail']}")
