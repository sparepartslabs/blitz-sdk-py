"""Minimal blitz example.

Install:
    pip install -e '.[anthropic]'   # or .[all] for all three providers

Run:
    export BLITZ_API_KEY=sk_...
    python examples/quickstart.py
"""

import os

import anthropic

import blitz

blitz.init(
    project_id="proj_demo",
    api_key=os.environ["BLITZ_API_KEY"],
    endpoint=os.environ.get("BLITZ_ENDPOINT", "http://localhost:8000"),
    sample_rate=1.0,  # trace everything in the demo
    environment=os.environ.get("BLITZ_ENV", "dev"),  # where these traces came from
)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=128,
    messages=[{"role": "user", "content": "In one sentence, what is distributed tracing?"}],
)

print(resp.content[0].text)
print("\n--> Open your blitz dashboard; this call is now a trace.")
