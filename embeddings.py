"""
Tier 2: Embedding similarity classification.
use semantic matching to detect what rules may miss
"""

import os
from openai import OpenAI #lets use openAi models for this step (they have a good embedding moddel)
import chromadb #chroma db
from dotenv import load_dotenv

load_dotenv() #our OpenAi key is stored in .env (added to git ignore)

client = OpenAI()

#build a small intent index
INTENT_EXAMPLES = {
    "billing": [
        "I was charged the wrong amount",
        "There's an unexpected charge on my account",
        "I think you took too much money from me",
        "I'd like a refund for my order",
        "Can I return this book?",
        "I want to send the book back",
        "You guys billed me twice for the same order",
    ],
    "shipping": [
        "I haven't received my books yet",
        "Where is my delivery?",
        "My order still hasn't shown up",
        "Can you tell me when my books will arrive?",
        "The tracking number isn't working",
        "My shipment seems to be stuck",
        "I've been waiting two weeks for my order",
    ],
    "account": [
        "I can't get into my account",
        "I forgot my login credentials",
        "I want to delete my account",
        "How do I change my password?",
        "I want to cancel my account",
        "My account seems to be locked",
        "I need to update my shipping address",
    ],
    "technical": [
        "The website isn't working for me",
        "I keep getting an error message",
        "Your site won't load",
        "My ebook won't download",
        "The book file won't open on my device",
        "I'm having trouble with the app",
        "Nothing happens when I click the button",
    ],
    "escalation": [
        "I want to speak with someone in charge",
        "I need to talk to a real person",
        "This is completely unacceptable",
        "I'm going to take legal action",
        "Get me your manager right now",
        "I'm extremely frustrated with your service",
        "I've been dealing with this for weeks",
    ],
}

#if were above this threshold, we can route it and choose a playbook
#0.85 used as a default, but we can change this according to the quality of our responses
CONFIDENCE_THRESHOLD = 0.85


class EmbeddingClassifier:
    """
    Builds an in-memory intent index at startup, then classifies
    incoming messages by embedding similarity at runtime.
    """

    def __init__(self):
        print("Building intent index...")
        self.collection = self._build_intent_index()
        print("Intent index ready.")

    def _build_intent_index(self) -> chromadb.Collection: 
        """
        Embed all example phrases and store them in Chroma.
        Uses a single batched API call — not one call per example.
        """
        # Flatten examples into parallel lists Chroma expects
        documents = []
        metadatas = []
        ids = []
        #loop through the intent examples index
        for intent, examples in INTENT_EXAMPLES.items():
            for i, example in enumerate(examples):
                documents.append(example)
                metadatas.append({"intent": intent})
                ids.append(f"{intent}_{i}")

        #use "text-embedding-3-small", a fast and inexpensive embedding model
        response = client.embeddings.create(
            input=documents,
            model="text-embedding-3-small",
        )
        embeddings = [item.embedding for item in response.data]

        # cosine space: similarity = 1 - distance, range [0, 1]
        chroma = chromadb.Client()
        collection = chroma.create_collection(
            name="intent_index",
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return collection

    def classify(self, message: str) -> dict:
        """
        Embed the incoming message and find the closest intent.
        Returns same dict shape as rules.py for consistency.
        """
        #embed the query from the customer
        response = client.embeddings.create(
            input=message,
            model="text-embedding-3-small",
        )
        query_embedding = response.data[0].embedding

        #query the index to find the top three most similar results
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
        )

        # Chroma cosine distance: 0 = identical, 1 = orthogonal
        # Convert to similarity: 1 - distance
        top_distance = results["distances"][0][0]
        similarity = 1 - top_distance
        top_intent = results["metadatas"][0][0]["intent"]
        top_example = results["documents"][0][0]

        if similarity >= CONFIDENCE_THRESHOLD:
            #same format used across each tier
            return {
                "intent": top_intent,
                "confidence": round(similarity, 4),
                "matched_example": top_example,
                "confident": True,
                "tier": "embeddings",
                "resolved": False,
            }

        #use tier 3 if under the threshold
        return {
            "intent": None,
            "confidence": round(similarity, 4),
            "confident": False,
            "tier": "embeddings",
            "resolved": False,
        }


# Quick test
if __name__ == "__main__":
    classifier = EmbeddingClassifier()

    test_messages = [
        "I think you guys took too much money from my card",   # no keyword match → billing
        "My package hasn't shown up and it's been 10 days",   # shipping
        "I can't remember my login info",                      # account
        "The button just spins and nothing happens",           # technical
        "I want to speak to someone senior",                   # escalation
        "What's the weather like today?",                      # should fail threshold
    ]

    print("\n" + "=" * 60)
    print("TIER 2: EMBEDDING SIMILARITY CLASSIFICATION")
    print("=" * 60)

    for msg in test_messages:
        result = classifier.classify(msg)
        print(f"\nMessage: {msg}")
        print(f"  Intent:     {result['intent']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Confident:  {result['confident']}")
        if result.get("matched_example"):
            print(f"  Matched:    {result['matched_example']}")
