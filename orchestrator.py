"""
Orchestrator: the main class that wires the full pipeline together.

This is the only file an external caller needs to interact with.
Everything else (rules, embeddings, classifier, router, RAG, response, validation)
is an implementation detail called from here.

Pipeline order:
  Tier 1 (rules) → Tier 2 (embeddings) → Tier 3 (LLM) → Router → RAG → Generate → Validate

Design principles:
  - Each tier only runs if the previous one didn't classify confidently
  - Tier 1 extracted data (order IDs, emails) is preserved through all tiers
  - Validation failure triggers one retry with a bigger model, then human escalation
  - Full trace is returned for every message — observability built in
"""

from rules import rule_based_check
from embeddings import EmbeddingClassifier
from classifier import llm_classify
from router import route
from knowledge_base import KnowledgeBase
from rag import retrieve
from response import generate
from validator import validate


# Canned escalation response — used when all automated resolution fails
ESCALATION_RESPONSE = (
    "I want to make sure you get the best help possible. "
    "Let me connect you with a human agent who can fully assist you."
)


class Agent:
    """
    The full customer support agent.
    Initialize once, call process() for each incoming message.
    """

    def __init__(self):
        # These two are built at startup because they require embedding API calls.
        # Everything else is a stateless function — no initialization needed.
        print("Initializing agent...")
        self.kb = KnowledgeBase()
        self.embedding_classifier = EmbeddingClassifier()
        print("Agent ready.\n")

    def process(self, message: str, conversation_history: list = None) -> dict:
        """
        Process a single customer message through the full pipeline.

        message:              the raw customer message string
        conversation_history: list of {"role": "user"/"assistant", "content": "..."} dicts
                              pass [] or omit for the first message in a session

        Returns a dict with:
          response:  the final text to send to the customer
          metadata:  full trace of what happened (for logging/observability)
        """
        if conversation_history is None:
            conversation_history = []

        # ----------------------------------------------------------------
        # STEP 1: Tier 1 — Rule-based classification
        # Always runs. Always extracts structured data regardless of intent match.
        # ----------------------------------------------------------------
        classification = rule_based_check(message)

        # Static response — fully resolved, no models needed
        if classification.get("resolved"):
            return {
                "response": classification["response"],
                "metadata": {
                    "classification_tier": "rules",
                    "intent": "static",
                    "resolved_at_tier": "rules",
                    "model_used": None,
                    "tools_called": [],
                    "retrieval_quality": None,
                    "validation_passed": True,
                    "validation_action": "send",
                    "iterations": None,
                    "escalated": False,
                    "sentiment": None,
                },
            }

        # ----------------------------------------------------------------
        # STEP 2: Tier 2 — Embedding similarity
        # Only runs if Tier 1 didn't classify confidently.
        # Preserves extracted data from Tier 1.
        # ----------------------------------------------------------------
        if not classification["confident"]:
            tier2 = self.embedding_classifier.classify(message)
            if tier2["confident"]:
                # Merge: take Tier 2's classification, keep Tier 1's extracted data
                classification = {**tier2, "extracted": classification.get("extracted", {})}

        # ----------------------------------------------------------------
        # STEP 3: Tier 3 — LLM classifier
        # Only runs if still not confident after Tier 2.
        # ----------------------------------------------------------------
        if not classification["confident"]:
            tier3 = llm_classify(message)
            # Merge: take Tier 3's richer output, keep Tier 1's extracted data
            classification = {**tier3, "extracted": classification.get("extracted", {})}

        # ----------------------------------------------------------------
        # STEP 4: Router
        # Maps classification result to a playbook — model, KB section, tools, tone.
        # Pure Python, no API calls.
        # ----------------------------------------------------------------
        playbook = route(classification)

        # ----------------------------------------------------------------
        # STEP 5: RAG retrieval
        # Scoped search of the knowledge base using the playbook's kb_section.
        # ----------------------------------------------------------------
        retrieval = retrieve(message, playbook, self.kb.collection)

        # ----------------------------------------------------------------
        # STEP 6: Response generation (agentic loop)
        # Model generates a response, optionally calling tools.
        # ----------------------------------------------------------------
        generation = generate(message, playbook, retrieval, conversation_history)

        # ----------------------------------------------------------------
        # STEP 7: Validation
        # Rule checks + hallucination check + grounding check.
        # ----------------------------------------------------------------
        validation = validate(generation["response"], retrieval, generation)

        if validation["action"] == "send":
            return self._build_result(
                response=generation["response"],
                classification=classification,
                playbook=playbook,
                retrieval=retrieval,
                generation=generation,
                validation=validation,
            )

        # ----------------------------------------------------------------
        # STEP 7a: Validation failed — retry with bigger model (one attempt)
        # ----------------------------------------------------------------
        if validation["action"] == "retry_bigger_model":
            playbook["model"] = "gpt-4o"
            retry_generation = generate(message, playbook, retrieval, conversation_history)
            retry_validation = validate(retry_generation["response"], retrieval, retry_generation)

            if retry_validation["action"] == "send":
                return self._build_result(
                    response=retry_generation["response"],
                    classification=classification,
                    playbook=playbook,
                    retrieval=retrieval,
                    generation=retry_generation,
                    validation=retry_validation,
                    retried=True,
                )

            # Retry also failed — fall through to human escalation

        # ----------------------------------------------------------------
        # STEP 7b: Escalate to human
        # Attach full context so the human agent has everything they need.
        # ----------------------------------------------------------------
        return self._build_result(
            response=ESCALATION_RESPONSE,
            classification=classification,
            playbook=playbook,
            retrieval=retrieval,
            generation=generation,
            validation=validation,
            escalated=True,
        )

    def _build_result(
        self,
        response: str,
        classification: dict,
        playbook: dict,
        retrieval: dict,
        generation: dict,
        validation: dict,
        retried: bool = False,
        escalated: bool = False,
    ) -> dict:
        """
        Build the final result dict.
        Consistent shape regardless of which path the pipeline took.
        The metadata here is your observability layer — log all of it in production.
        """
        return {
            "response": response,
            "metadata": {
                "classification_tier": classification.get("tier"),
                "intent": classification.get("intent"),
                "sentiment": classification.get("sentiment"),
                "complexity": classification.get("complexity"),
                "model_used": generation.get("model"),
                "tools_called": generation.get("tools_called", []),
                "retrieval_quality": retrieval.get("quality"),
                "retrieval_scope": retrieval.get("search_scope"),
                "validation_passed": validation.get("passed"),
                "validation_action": validation.get("action"),
                "validation_issues": validation.get("issues", []),
                "grounding_score": validation.get("grounding_score"),
                "iterations": generation.get("iterations"),
                "retried": retried,
                "escalated": escalated,
                "extracted": classification.get("extracted", {}),
            },
        }

    def chat(self):
        """
        Simple interactive loop for manual testing in the terminal.
        Maintains conversation history across turns.
        """
        print("Agent ready. Type 'quit' to exit.\n")
        conversation_history = []

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            result = self.process(user_input, conversation_history)
            print(f"\nAgent: {result['response']}")
            print(f"[tier={result['metadata']['classification_tier']} | "
                  f"intent={result['metadata']['intent']} | "
                  f"sentiment={result['metadata']['sentiment']} | "
                  f"model={result['metadata']['model_used']} | "
                  f"tools={[t['tool'] for t in result['metadata']['tools_called']]} | "
                  f"quality={result['metadata']['retrieval_quality']} | "
                  f"iterations={result['metadata']['iterations']} | "
                  f"valid={result['metadata']['validation_passed']} | "
                  f"escalated={result['metadata']['escalated']}]\n")

            # Update conversation history for next turn
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": result["response"]})


if __name__ == "__main__":
    agent = Agent()
    agent.chat()
