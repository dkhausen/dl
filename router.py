"""
Router: Intent-to-playbook mapping.
Pure Python — no models, no API calls, no cost.

Use the classification result to call a playbook
"""


# Playbooks define behavior per intent.
# Each key maps to a config that downstream components read.
#
# model:           which LLM generates the final response
# kb_section:      which section of the vector DB to search (scoped retrieval)
# tools:           which tools the agent is allowed to call
# tone:            instruction injected into the system prompt
# response_length: hint for how verbose the response should be
# escalate:        whether this intent should go to a human by default
PLAYBOOKS = {
    "billing": {
        "model": "gpt-4o-mini",
        "kb_section": "billing",
        "tools": ["billing_lookup"],
        "tone": "empathetic and solution-focused",
        "response_length": "medium",
        "escalate": False,
    },
    "shipping": {
        "model": "gpt-4o-mini",
        "kb_section": "shipping",
        "tools": ["order_lookup"],
        "tone": "reassuring and informative",
        "response_length": "medium",
        "escalate": False,
    },
    "account": {
        "model": "gpt-4o-mini",
        "kb_section": "account",
        "tools": ["account_check"],
        "tone": "clear and helpful",
        "response_length": "short",
        "escalate": False,
    },
    "technical": {
        "model": "gpt-4o",           # technical issues need better reasoning
        "kb_section": "technical",
        "tools": [],
        "tone": "technical but accessible",
        "response_length": "long",   # tech issues often need step-by-step guidance
        "escalate": False,
    },
    "escalation": {
        "model": "gpt-4o",           # high stakes — use the best model
        "kb_section": "general",
        "tools": ["account_check"],
        "tone": "deeply empathetic and de-escalating",
        "response_length": "medium",
        "escalate": True,            # flag for potential human handoff
    }, # we have an unknown in case the tier 3 classifier makes a mistake or we truly can't classify the comment
    "unknown": {
        "model": "gpt-4o-mini",
        "kb_section": "general",
        "tools": [],
        "tone": "friendly and helpful",
        "response_length": "short",
        "escalate": False,
    },
}

# Sentiment overrides — applied on top of the base playbook.
# An angry customer always gets the best model and a human escalation flag,
# regardless of what the intent playbook says.
SENTIMENT_OVERRIDES = {
    "angry": {
        "model": "gpt-4o",
        "escalate": True,
    }
}

# More complex = more chunks that the RAG layer pulls
COMPLEXITY_CHUNK_MAP = {
    "simple": 3,
    "moderate": 5,
    "complex": 8,
}

# default chunks when Tier 3 doesn't return it
DEFAULT_CHUNK_COUNT = 5


def route(classification: dict) -> dict:
    """
    Takes a classification result from any tier and returns a playbook.
    Applies sentiment and complexity overrides on top of the base playbook.
    """
    intent = classification.get("intent") or "unknown"

    # look up the base playbook — fall back to "unknown" if intent isn't mapped. copy makes sure we don't change the original PLAYBOOKS dict
    playbook = PLAYBOOKS.get(intent, PLAYBOOKS["unknown"]).copy() 

    # apply sentiment override if present (Tier 3 provides this, Tier 1/2 don't)
    sentiment = classification.get("sentiment")
    if sentiment and sentiment in SENTIMENT_OVERRIDES:
        playbook.update(SENTIMENT_OVERRIDES[sentiment])

    # attach chunk count based on complexity signal
    complexity = classification.get("complexity", "moderate")
    playbook["chunk_count"] = COMPLEXITY_CHUNK_MAP.get(complexity, DEFAULT_CHUNK_COUNT)

    # pass through useful classification fields for downstream use
    playbook["intent"] = intent
    playbook["sentiment"] = sentiment
    playbook["complexity"] = complexity
    playbook["suggested_tool"] = classification.get("suggested_tool")
    playbook["extracted"] = classification.get("extracted", {})
    playbook["tier"] = classification.get("tier")

    return playbook


# Quick test
if __name__ == "__main__":
    test_cases = [
        # Tier 1 result — no sentiment or complexity
        {
            "intent": "billing",
            "confident": True,
            "tier": "rules",
            "resolved": False,
            "extracted": {"order_id": "ORD-12345"},
        },
        # Tier 3 result — angry customer with billing issue
        {
            "intent": "billing",
            "confidence": 0.95,
            "sentiment": "angry",
            "complexity": "complex",
            "suggested_tool": "billing_lookup",
            "confident": True,
            "tier": "llm",
            "resolved": False,
        },
        # Tier 3 result — simple account question
        {
            "intent": "account",
            "confidence": 0.92,
            "sentiment": "neutral",
            "complexity": "simple",
            "suggested_tool": "account_check",
            "confident": True,
            "tier": "llm",
            "resolved": False,
        },
        # Tier 3 result — escalation intent
        {
            "intent": "escalation",
            "confidence": 0.97,
            "sentiment": "angry",
            "complexity": "complex",
            "suggested_tool": None,
            "confident": True,
            "tier": "llm",
            "resolved": False,
        },
    ]

    print("=" * 60)
    print("ROUTER: INTENT TO PLAYBOOK")
    print("=" * 60)

    for classification in test_cases:
        playbook = route(classification)
        print(f"\nIntent:     {classification['intent']} "
              f"(sentiment: {classification.get('sentiment', 'n/a')}, "
              f"complexity: {classification.get('complexity', 'n/a')})")
        print(f"  Model:       {playbook['model']}")
        print(f"  KB section:  {playbook['kb_section']}")
        print(f"  Tools:       {playbook['tools']}")
        print(f"  Chunks:      {playbook['chunk_count']}")
        print(f"  Escalate:    {playbook['escalate']}")
        print(f"  Tone:        {playbook['tone']}")
