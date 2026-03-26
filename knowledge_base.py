"""
Knowledge base: sample documents and index builder.
Content would come from a company's actual docs in production, generated here.
Each piece of content has a tag to make scoping more efficient
"""

import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Sample knowledge base documents organized by section.
# Structure: list of dicts with "text" and "section".
# In production: each chunk would also carry a source URL, doc ID, last-updated date.
KB_DOCUMENTS = [

    # --- BILLING ---
    {
        "text": "Bookly accepts returns within 30 days of delivery. Books must be in their original condition — unread and undamaged. To start a return, contact support with your order ID and we will provide a prepaid return label.",
        "section": "billing",
    },
    {
        "text": "Refunds are issued to the original payment method within 5-7 business days after we receive the returned item. You will receive an email confirmation when the refund is processed.",
        "section": "billing",
    },
    {
        "text": "If you were charged twice for the same order, this is usually caused by a payment retry after a failed transaction. Contact support with your order ID and email address and we will reverse the duplicate charge immediately.",
        "section": "billing",
    },
    {
        "text": "Digital ebook purchases are non-refundable once downloaded. If you have not yet downloaded the ebook, contact support within 7 days of purchase for a full refund.",
        "section": "billing",
    },
    {
        "text": "If your payment fails at checkout, double-check that your card details are correct and your billing address matches what your bank has on file. If the problem persists, try a different payment method or contact your bank.",
        "section": "billing",
    },

    # --- SHIPPING ---
    {
        "text": "Standard shipping takes 5-7 business days. Expedited shipping (2-3 business days) is available at checkout for $8.99. Overnight shipping is available for $19.99 on orders placed before 2pm EST.",
        "section": "shipping",
    },
    {
        "text": "Once your order ships, you will receive a tracking number via email. You can also track your order at any time by visiting the Orders section of your Bookly account.",
        "section": "shipping",
    },
    {
        "text": "If your order has not arrived within 10 business days of the estimated delivery date, contact support with your order ID. We will investigate with the carrier and either reship the books or issue a full refund.",
        "section": "shipping",
    },
    {
        "text": "Bookly ships to all 50 US states and over 40 countries internationally. International orders may be subject to customs fees, which are the responsibility of the recipient. Delivery times for international orders are 10-21 business days.",
        "section": "shipping",
    },
    {
        "text": "If your tracking shows delivered but you have not received the package, first check with neighbors and your building's front desk or mail room. If still missing after 24 hours, contact support and we will file a carrier claim on your behalf.",
        "section": "shipping",
    },

    # --- ACCOUNT ---
    {
        "text": "To reset your password, click 'Forgot Password' on the Bookly login page and enter your email address. You will receive a reset link within 5 minutes. Check your spam folder if it does not arrive.",
        "section": "account",
    },
    {
        "text": "To cancel your Bookly account, go to Account Settings > Manage Account > Close Account. Any pending orders will still be fulfilled. You will receive a confirmation email once your account is closed.",
        "section": "account",
    },
    {
        "text": "If your account has been locked due to too many failed login attempts, wait 30 minutes and try again. If you are still locked out, contact support and we will manually unlock the account after verifying your identity.",
        "section": "account",
    },
    {
        "text": "To update your email address or shipping address, go to Account Settings > Profile. Email changes require verification — you will receive a confirmation link at your new address before the change takes effect.",
        "section": "account",
    },
    {
        "text": "Bookly's wishlist feature lets you save books for later. To create a wishlist, click the heart icon on any book page. Wishlists are private by default but can be shared via a link from your account settings.",
        "section": "account",
    },

    # --- TECHNICAL ---
    {
        "text": "If the Bookly website is not loading or displaying correctly, try clearing your browser cache and cookies, or open an incognito window. Bookly supports Chrome, Firefox, Safari, and Edge.",
        "section": "technical",
    },
    {
        "text": "If you are having trouble downloading a purchased ebook, try logging out and back in, then visit My Library and click Download again. Make sure you have a compatible ebook reader app installed (Kindle, Apple Books, or Adobe Digital Editions).",
        "section": "technical",
    },
    {
        "text": "If the Bookly mobile app is crashing, try force-closing the app and reopening it. If the issue persists, uninstall and reinstall the app. Make sure you are running the latest version — check your device's app store for updates.",
        "section": "technical",
    },
    {
        "text": "Bookly gift cards can be redeemed at checkout by entering the gift card code in the 'Promo/Gift Card' field. Gift cards do not expire and cannot be exchanged for cash. If your gift card code is not working, contact support.",
        "section": "technical",
    },
    {
        "text": "Two-factor authentication (2FA) can be enabled in Account Settings > Security. Bookly supports authenticator apps (Google Authenticator, Authy) and SMS codes. If you lose access to your 2FA device, contact support to recover your account.",
        "section": "technical",
    },

    # --- GENERAL ---
    {
        "text": "Bookly's customer support team is available Monday through Friday, 9am to 6pm EST. You can reach us at support@bookly.com or through the chat on our website. We aim to respond to all inquiries within 24 hours.",
        "section": "general",
    },
    {
        "text": "Bookly takes data privacy seriously. All customer data is encrypted at rest and in transit. We do not sell customer data to third parties. You can request a copy of your data or account deletion at any time by contacting support.",
        "section": "general",
    },
    {
        "text": "Bookly offers a price match guarantee. If you find a lower price on a physical book at a major retailer within 7 days of your purchase, contact support with a link to the lower price and we will refund the difference.",
        "section": "general",
    },
]


class KnowledgeBase:
    """
    Embeds KB documents and stores them in Chroma at startup.
    Exposes the collection for the RAG retrieval layer to query.
    """

    def __init__(self):
        print("Building knowledge base index...")
        self.collection = self._build_index()
        print(f"Knowledge base ready ({len(KB_DOCUMENTS)} documents).")

    def _build_index(self) -> chromadb.Collection:
        """
        Embed all KB documents in a single batched API call and store in Chroma.
        Each document is tagged with its section for scoped retrieval.
        """
        texts = [doc["text"] for doc in KB_DOCUMENTS]
        metadatas = [{"section": doc["section"]} for doc in KB_DOCUMENTS]
        ids = [f"doc_{i}" for i in range(len(KB_DOCUMENTS))]

        # One API call for all documents
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
        )
        embeddings = [item.embedding for item in response.data]

        chroma = chromadb.Client()
        collection = chroma.create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return collection
