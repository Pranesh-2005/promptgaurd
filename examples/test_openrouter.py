"""promptgaurd + OpenRouter (OpenAI-compatible endpoint). Needs OPENROUTER_API_KEY.

Uses the openai SDK pointed at OpenRouter's endpoint - no extra dependency.
Logs go to examples/logs/openrouter.log - the console only shows responses.
"""

import os

from openai import OpenAI

from promptgaurd import guard_client
from _common import ATTACK, BENIGN, load_dotenv, make_gaudrial, require_env, show

load_dotenv()
require_env("OPENROUTER_API_KEY")

MODEL = "openai/gpt-4o-mini"  # any OpenRouter model id
client = guard_client(
    OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]),
    gaudrial=make_gaudrial("openrouter"),
    provider="openrouter",
)


def ask(label, prompt):
    r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
    show(label, r.choices[0].message.content)


ask("BENIGN", BENIGN)   # real OpenRouter answer
ask("ATTACK", ATTACK)   # block template - OpenRouter never called
