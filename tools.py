"""
Mock tools: order_lookup, account_check, billing_lookup.

These would call internal APIs in production

Includes tool schemas in a structured definition Open AI needs to know
and tool implementations, the python functions that execute


"""


# --- TOOL SCHEMAS ---
# OpenAI function calling requires each tool to be described as a JSON schema.
# The model reads these schemas and decides when and how to call each tool.
# It never calls Python directly — it returns a structured tool_call object,
# and our orchestration layer executes the actual function.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "order_lookup",
            "description": "Look up the current status and details of a customer order by order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID, e.g. ORD-12345",
                    }
                },
                "required": ["order_id"], # requires the model to get order_id from the customer 
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "account_check",
            "description": "Look up a customer's account status, plan, and recent activity by email address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The customer's email address",
                    }
                },
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "billing_lookup",
            "description": "Look up recent charges and billing history for a customer account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The customer's email address",
                    },
                    "order_id": {
                        "type": "string",
                        "description": "Optional order ID to scope the lookup",
                    },
                },
                "required": ["email"], # order ID is not required here, we can use the customer's email
            },
        },
    },
]


# --- TOOL IMPLEMENTATIONS ---
# These are mock implementations. In production, replace the return values
# with actual API calls to internal systems.

def order_lookup(order_id: str) -> dict:
    """Mock order lookup — returns realistic fake data."""
    
    # actual API call would look like this
                                                                        
    # production: replace with internal API call
    # response = requests.get(f"https://api.bookly.com/orders/{order_id}", headers=AUTH_HEADERS)             
    # return response.json()   

    mock_orders = {
        "ORD-12345": {
            "status": "in_transit",
            "carrier": "UPS",
            "tracking_number": "1Z999AA10123456784",
            "estimated_delivery": "2026-03-27",
            "items": ["The Midnight Library by Matt Haig", "Atomic Habits by James Clear"],
            "shipped_date": "2026-03-21",
        },
        "ORD-789": {
            "status": "delivered",
            "carrier": "FedEx",
            "tracking_number": "794644792798",
            "delivered_date": "2026-03-20",
            "items": ["Dune by Frank Herbert"],
        },
    }
    if order_id in mock_orders:
        return {"found": True, "order": mock_orders[order_id]}
    return {"found": False, "message": f"No order found with ID {order_id}"}


def account_check(email: str) -> dict:
    """Mock account lookup — returns realistic fake data."""
    return {
        "found": True,
        "account": {
            "email": email,
            "plan": "Pro",
            "status": "active",
            "member_since": "2024-06-01",
            "last_login": "2026-03-21",
            "billing_day": 1,
        },
    }


def billing_lookup(email: str, order_id: str = None) -> dict:
    """Mock billing lookup — returns recent charges."""
    charges = [
        {"date": "2026-03-21", "amount": "$34.97", "description": "Order ORD-12345 - 2 books", "status": "paid"},
        {"date": "2026-03-21", "amount": "$34.97", "description": "Order ORD-12345 - 2 books", "status": "paid"},  # duplicate
        {"date": "2026-03-10", "amount": "$14.99", "description": "Order ORD-789 - Dune by Frank Herbert", "status": "paid"},
    ]
    return {
        "found": True,
        "email": email,
        "recent_charges": charges,
        "note": "Possible duplicate charge detected on 2026-03-21 for order ORD-12345",
    }


# Dispatcher — maps tool name strings to actual functions.
# The orchestration layer calls this instead of importing each function individually.
TOOL_REGISTRY = {
    "order_lookup": order_lookup,
    "account_check": account_check,
    "billing_lookup": billing_lookup,
}


def execute_tool(name: str, arguments: dict) -> dict:
    """
    Execute a tool by name with the given arguments.
    Called by the agentic loop when the model returns a tool_call.
    """
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}
    return TOOL_REGISTRY[name](**arguments) # unpacks the dict into keyword arguments e.g. order_lookup(order_id="ORD-12345")  
