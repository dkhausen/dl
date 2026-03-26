"""
Response generation: dynamic prompts, tool use, agentic loop.

This is where everything upstream comes together:

brings everything together:
1) playbook
2) retrieval
3) extracted data
4) conversation history
5) tools

houses the agentic loop
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from tools import TOOL_SCHEMAS, execute_tool

load_dotenv()

client = OpenAI()

MAX_ITERATIONS = 5  # safety cap on agentic loop — prevents infinite tool-calling


def _build_system_prompt(playbook: dict, retrieval: dict) -> str:
    """
    Build the system prompt dynamically from the playbook and retrieval results.
    Every response gets a different system prompt based on intent, tone, and context.
    """
    tone = playbook.get("tone", "professional and helpful")
    response_length = playbook.get("response_length", "medium")
    escalate = playbook.get("escalate", False)
    suggested_tool = playbook.get("suggested_tool")
    retrieval_quality = retrieval.get("quality", "high")
    chunks = retrieval.get("chunks", [])

    # Format KB chunks into readable context block
    if chunks:
        context_block = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)
        )
    else:
        context_block = "No relevant knowledge base content was found."

    # Retrieval quality warning — tells the model to be honest when context is weak
    if retrieval_quality == "low":
        context_warning = (
            "\nWARNING: Retrieval confidence was low. The context above may not be "
            "sufficient to fully answer this question. If you cannot answer confidently "
            "from the context provided, say so clearly and offer to escalate to a human agent."
        )
    elif retrieval_quality == "medium":
        context_warning = (
            "\nNOTE: Retrieval confidence was moderate. Use the context provided but "
            "acknowledge uncertainty if the answer is not clearly supported."
        )
    else:
        context_warning = ""

    # Escalation instruction — injected when playbook or sentiment flags it
    escalation_instruction = ""
    if escalate:
        escalation_instruction = (
            "\nIMPORTANT: This customer may need to speak with a human agent. "
            "After addressing their immediate concern, offer to connect them with a specialist."
        )

    # Suggested tool hint — soft signal from Tier 3, model decides whether to use it
    tool_hint = ""
    if suggested_tool:
        tool_hint = f"\nSuggested tool: Consider using the {suggested_tool} tool if it would help resolve this request."

    prompt = f"""You are a customer support agent for Bookly, an online bookstore.

Tone: {tone}
Response length: {response_length} (short = 1-2 sentences, medium = 1 paragraph, long = multiple paragraphs if needed)

KNOWLEDGE BASE CONTEXT:
{context_block}
{context_warning}

CAPABILITIES AND LIMITATIONS:
- You CAN look up order status, billing history, and account details using tools
- You CANNOT process refunds, issue credits, cancel orders, or modify accounts
- If a customer needs a refund or wants to cancel an order, explain the policy and let them know a human agent will action it — never imply you have done it yourself

RULES:
- Only answer based on the knowledge base context provided above
- Never invent specific numbers, dates, policies, or order details
- Never mention competitor products or stores
- If a customer asks about a specific order but has not provided an order ID, ask for it before proceeding
- If a customer needs account or billing help but has not provided their email, ask for it before proceeding
- If a customer asks multiple questions, address each one clearly
- Always be honest if you don't have enough information to fully resolve the issue
{escalation_instruction}
{tool_hint}"""

    return prompt.strip()


def generate(
    message: str,
    playbook: dict,
    retrieval: dict,
    conversation_history: list,
) -> dict:
    """
    Generate a response using the agentic loop.

    conversation_history: list of {"role": "user"/"assistant", "content": "..."} dicts
    Returns a dict with the final response text and metadata about what happened.
    """
    model = playbook.get("model", "gpt-4o-mini")
    extracted = playbook.get("extracted", {})

    # Build the dynamic system prompt
    system_prompt = _build_system_prompt(playbook, retrieval)

    # Construct the full message list:
    # [system] + [history] + [current user message]
    messages = [{"role": "system", "content": system_prompt}] # system prompt we constructed 
    messages.extend(conversation_history) # include conversation history to help with context
    messages.append({"role": "user", "content": message}) # what the user said

    # If Tier 1 extracted structured data, append it as a system note.
    # This pre-fills context so the model can use it for tool calls without
    # asking the customer to repeat themselves.
    if extracted:
        extracted_note = "Extracted from customer message: " + json.dumps(extracted)
        messages.append({"role": "system", "content": extracted_note})

    # --- AGENTIC LOOP ---
    tools_called = []

    # Scope tools to what this intent's playbook allows — not all tools are available to all intents
    allowed_tools = playbook.get("tools", [])
    scoped_tools = [t for t in TOOL_SCHEMAS if t["function"]["name"] in allowed_tools]

    for iteration in range(MAX_ITERATIONS): # when we hit 5 iterations, it's unlikely we'll get a correct answer without a human
        # Only pass tools/tool_choice when this intent has permitted tools
        # OpenAI throws a 400 if tool_choice is set without tools
        call_kwargs = {"model": model, "temperature": 0.3, "messages": messages}
        if scoped_tools:
            call_kwargs["tools"] = scoped_tools
            call_kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**call_kwargs)

        choice = response.choices[0]

        # Model is done — return the final text response
        if choice.finish_reason == "stop":
            return {
                "response": choice.message.content,
                "model": model,
                "tools_called": tools_called,
                "iterations": iteration + 1,
                "retrieval_quality": retrieval.get("quality"),
            }

        # Model wants to call one or more tools
        if choice.finish_reason == "tool_calls":
            # Add the model's tool_call message to history so it has context
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments) #convert JSON string so we can pass it

                # Execute the tool and record it
                result = execute_tool(name, args)
                tools_called.append({"tool": name, "args": args, "result": result})

                # Feed the tool result back into the message list
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id, # map tool result back to tool
                    "content": json.dumps(result),
                })

            # Loop continues — model will now generate a response using the tool results

    # Hit the iteration cap — force a safe fallback
    return {
        "response": "I wasn't able to fully resolve your issue. Let me connect you with a human agent who can help.",
        "model": model,
        "tools_called": tools_called,
        "iterations": MAX_ITERATIONS,
        "retrieval_quality": retrieval.get("quality"),
        "escalated": True,
    }


# Quick test
if __name__ == "__main__":
    from knowledge_base import KnowledgeBase
    from rag import retrieve

    kb = KnowledgeBase()

    test_cases = [
        {
            "message": "I was charged twice for order ORD-12345. My email is test@example.com",
            "playbook": {
                "intent": "billing",
                "model": "gpt-4o-mini",
                "kb_section": "billing",
                "chunk_count": 3,
                "tone": "empathetic and solution-focused",
                "response_length": "medium",
                "escalate": False,
                "suggested_tool": "billing_lookup",
                "extracted": {"order_id": "ORD-12345", "email": "test@example.com"},
            },
        },
        {
            "message": "Where is my order ORD-789?",
            "playbook": {
                "intent": "shipping",
                "model": "gpt-4o-mini",
                "kb_section": "shipping",
                "chunk_count": 3,
                "tone": "reassuring and informative",
                "response_length": "medium",
                "escalate": False,
                "suggested_tool": "order_lookup",
                "extracted": {"order_id": "ORD-789"},
            },
        },
    ]

    print("=" * 60)
    print("RESPONSE GENERATION")
    print("=" * 60)

    for case in test_cases:
        retrieval = retrieve(case["message"], case["playbook"], kb.collection)
        result = generate(case["message"], case["playbook"], retrieval, [])

        print(f"\nMessage: {case['message']}")
        print(f"  Model:      {result['model']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Tools:      {[t['tool'] for t in result['tools_called']]}")
        print(f"  Response:\n    {result['response']}")
