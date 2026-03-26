"""
Tier 3: Lightweight LLM classifier.
runs when Tier 1 and Tier 2 fail. Uses chatgpt mini
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# The system prompt:
# 1) same JSON schema as before
# 2) provides few-shot examples
# combination of these two lead to predictable answers

SYSTEM_PROMPT = """You are an intent classifier for a customer support agent.
Classify the customer message and return a JSON object with exactly these fields:

{
  "intent": one of ["billing", "shipping", "account", "technical", "escalation", "unknown"],
  "confidence": float between 0.0 and 1.0,
  "sentiment": one of ["positive", "neutral", "negative", "angry"],
  "complexity": one of ["simple", "moderate", "complex"],
  "suggested_tool": one of ["order_lookup", "account_check", "billing_lookup", null]
}

Rules:
- Use "unknown" only if the message is truly unrelated to customer support
- "angry" sentiment means hostile or threatening tone, not just frustrated
- "complexity" reflects how much reasoning/context the response will need
- "suggested_tool" is the most likely tool needed, or null if none

Examples:

Message: "I think I was charged twice for my order"
Response: {"intent": "billing", "confidence": 0.97, "sentiment": "negative", "complexity": "moderate", "suggested_tool": "billing_lookup"}

Message: "I want to return the book I bought last week"
Response: {"intent": "billing", "confidence": 0.95, "sentiment": "neutral", "complexity": "moderate", "suggested_tool": "order_lookup"}

Message: "do you sell audiobooks"
Response: {"intent": "unknown", "confidence": 0.85, "sentiment": "neutral", "complexity": "simple", "suggested_tool": null}

Message: "I have been trying to fix this for THREE WEEKS and nobody helps me. I'm done."
Response: {"intent": "escalation", "confidence": 0.95, "sentiment": "angry", "complexity": "complex", "suggested_tool": "account_check"}

Message: "my account password isn't working"
Response: {"intent": "account", "confidence": 0.96, "sentiment": "neutral", "complexity": "simple", "suggested_tool": "account_check"}

Message: "the ebook I downloaded won't open on my Kindle"
Response: {"intent": "technical", "confidence": 0.94, "sentiment": "negative", "complexity": "moderate", "suggested_tool": null}

Message: "I ordered some books two weeks ago and they still haven't arrived"
Response: {"intent": "shipping", "confidence": 0.96, "sentiment": "negative", "complexity": "moderate", "suggested_tool": "order_lookup"}

Only return the JSON object. No explanation, no markdown, no extra text."""


def llm_classify(message: str) -> dict:
    """
    Classify the message using GPT-4o-mini
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,           # always temp of 0 for deterministic output
        response_format={"type": "json_object"},  # forces valid JSON output
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    return {
        "intent": parsed["intent"],
        "confidence": parsed["confidence"],
        "sentiment": parsed["sentiment"],
        "complexity": parsed["complexity"],
        "suggested_tool": parsed["suggested_tool"],
        "confident": parsed["confidence"] >= 0.7,  # tier 3 is the last gate, so return regardless
        "tier": "llm",
        "resolved": False,
    }


# Quick test
if __name__ == "__main__":
    test_messages = [
        "I was charged $49.99 but my plan is only $29.99",
        "This is absolutely ridiculous, I want to speak to a lawyer",
        "The app crashes every time I try to upload a file",
        "Do you offer discounts for students?",
        "I've been locked out of my account for two days",
    ]

    print("=" * 60)
    print("TIER 3: LLM CLASSIFIER")
    print("=" * 60)

    for msg in test_messages:
        result = llm_classify(msg)
        print(f"\nMessage: {msg}")
        print(f"  Intent:     {result['intent']} ({result['confidence']})")
        print(f"  Sentiment:  {result['sentiment']}")
        print(f"  Complexity: {result['complexity']}")
        print(f"  Tool:       {result['suggested_tool']}")
        print(f"  Tier:       {result['tier']}")
