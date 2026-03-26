# Bookly Customer Support Agent

A tiered customer support agent for Bookly, a fictional online bookstore. Built in Python using OpenAI and ChromaDB.

## Requirements

- Python 3.10+
- An OpenAI API key

## Setup

**1. Install dependencies**
```bash
pip install openai chromadb python-dotenv
```

**2. Add your OpenAI API key**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
```

Your key should start with `sk-`. You can generate one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

**3. Run the agent**
```bash
python orchestrator.py
```

This starts an interactive chat loop in the terminal. Type `quit` to exit.

## File Structure

```
orchestrator.py      Main agent — start here
rules.py             Tier 1: keyword classification + regex extraction
embeddings.py        Tier 2: embedding similarity classification
classifier.py        Tier 3: LLM intent + sentiment + complexity classification
router.py            Intent-to-playbook mapping
knowledge_base.py    KB documents + Chroma index
rag.py               Scoped retrieval with quality checks
response.py          Agentic loop + dynamic prompt builder
validator.py         Guardrails + grounding checks
tools.py             Tool schemas + mock implementations
```

## Notes

- The agent uses `gpt-4o-mini` by default and upgrades to `gpt-4o` for technical issues, angry customers, and validation retries
- All tool integrations (`order_lookup`, `account_check`, `billing_lookup`) are mocked with realistic fake data
- The `.env` file is gitignored — never commit your API key
