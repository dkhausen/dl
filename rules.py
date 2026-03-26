"""
Tier 1: Rule-based classification and data extraction
this is the first gate we can use to classify intent (and extract some free data along the way)
"""

import re


#a few basic static_responses that can finish the loop immediately if needed
STATIC_RESPONSES = {
    "what are your hours": "Our support team is available Monday through Friday, 9am to 6pm EST.",
    "how do i contact support": "You can reach us at support@bookly.com or use the chat on our website.",
    "do you ship internationally": "Yes, Bookly ships to over 40 countries. International orders may be subject to customs fees.",
    "what is your return policy": "Bookly accepts returns within 30 days of delivery. Books must be in original condition. Contact support to start a return.",
}

#keywords we can use to classify intent, if these are extracted by keyword_classify, intent is found since its deterministic.
#an edge case here would be if the user asks multiple questions at once and we identify >1 intent; more in the doc on how we may 
#want to handle this
KEYWORD_RULES = [
    {
        "keywords": ["charged twice", "double charged", "overcharged",
                     "extra charge", "wrong charge", "billing error"],
        "intent": "billing",
    },
    {
        "keywords": ["refund", "money back", "reimburse", "credit back",
                     "return", "send it back", "return my order", "exchange"],
        "intent": "billing",
    },
    {
        "keywords": ["track my order", "where's my order", "where is my order",
                     "shipping status", "delivery status", "when will it arrive",
                     "hasn't arrived", "not delivered", "where are my books"],
        "intent": "shipping",
    },
    {
        "keywords": ["reset my password", "forgot my password", "can't log in",
                     "cant log in", "locked out", "can't sign in"],
        "intent": "account",
    },
    {
        "keywords": ["cancel my order", "cancel my account",
                     "close my account", "delete my account"],
        "intent": "account",
    },
    {
        "keywords": ["not working", "broken", "bug", "error", "crash",
                     "keeps crashing", "won't load", "won't open"],
        "intent": "technical",
    },
    {
        "keywords": ["speak to a manager", "talk to a human", "supervisor",
                     "lawyer", "legal", "sue", "attorney"],
        "intent": "escalation",
    },
]

# Regex patterns for extracting structured data
EXTRACTION_PATTERNS = {
    "order_id": r'ORD-\d{3,6}',
    "email": r'[\w.-]+@[\w.-]+\.\w+',
    "phone": r'\d{3}[-.]?\d{3}[-.]?\d{4}',
    "dollar_amount": r'\$\d+\.?\d*',
}

def extract_data(message: str) -> dict:
    """Extract structured data from the message using regex."""
    extracted = {}
    for field, pattern in EXTRACTION_PATTERNS.items():
        match = re.search(pattern, message)
        if match:
            extracted[field] = match.group()
    return extracted

#can we get a quick question answer here before escalating to tier 2 and tier 3 classification methods?
def check_static_response(message: str) -> dict | None:
    """Check if the message matches a static response exactly."""
    message_lower = message.lower().strip()
    
#use a predictable pattern format here
    for pattern, response in STATIC_RESPONSES.items():
        if pattern in message_lower:
            return {
                "intent": "static",
                "response": response,
                "confident": True,
                "tier": "rules",
                "resolved": True,
            }
    return None

#we couldn't get a quick answer, so now were using Tier 1 to extract intent in a deterministic way
def keyword_classify(message: str) -> dict | None:
    """Classify intent using keyword matching."""
    message_lower = message.lower()
    
    for rule in KEYWORD_RULES:
        for keyword in rule["keywords"]:
            if re.search(r'\b' + re.escape(keyword) + r'\b', message_lower):
                return {
                    "intent": rule["intent"],
                    "matched_keyword": keyword,
                    "confident": True,
                    "tier": "rules",
                    "resolved": False,
                }
    return None


def rule_based_check(message: str) -> dict:
    """
    Run the full rule-based tier.
    Returns classification result and any extracted data.
    """
    # extract data because its free
    extracted = extract_data(message)
    
    # check static responses for quick support
    static = check_static_response(message)
    if static:
        static["extracted"] = extracted
        return static
    
    # can we get intent from a keyword in the question
    keyword_result = keyword_classify(message)
    if keyword_result:
        keyword_result["extracted"] = extracted
        return keyword_result
    
    # if not, pass down the intent anyways
    return {
        "intent": None,
        "confident": False,
        "tier": "rules",
        "resolved": False,
        "extracted": extracted,
    }


# Claude generated test
if __name__ == "__main__":
    test_messages = [
        "What are your hours?",
        "I was charged twice for order ORD-12345",
        "Where's my order ORD-789?",
        "I need help with something random",
        "I want to speak to a manager right now",
        "My app keeps crashing when I open it",
    ]
    
    print("=" * 60)
    print("TIER 1: RULE-BASED CLASSIFICATION")
    print("=" * 60)
    
    for msg in test_messages:
        result = rule_based_check(msg)
        print(f"\nMessage: {msg}")
        print(f"  Intent: {result['intent']}")
        print(f"  Confident: {result['confident']}")
        print(f"  Resolved: {result['resolved']}")
        if result.get('extracted'):
            print(f"  Extracted: {result['extracted']}")
        if result.get('response'):
            print(f"  Response: {result['response']}")
