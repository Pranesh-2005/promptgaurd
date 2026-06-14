"""promptguard + Groq (OpenAI-compatible endpoint). Needs GROQ_API_KEY.

Uses the openai SDK pointed at Groq's endpoint - no extra dependency.
Logs go to examples/logs/groq.log - the console only shows responses.
"""

import os

from openai import OpenAI

from promptguard import guard_client
from _common import ATTACK, BENIGN, load_dotenv, make_Guardial, require_env, show

load_dotenv()
require_env("GROQ_API_KEY")

MODEL = "llama-3.3-70b-versatile"
client = guard_client(
    OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"]),
    Guardial=make_Guardial("groq"),
    provider="groq",
)


def ask(label, prompt):
    r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
    show(label, r.choices[0].message.content)


ask("BENIGN", BENIGN)   # real Groq answer
ask("ATTACK", ATTACK)   # block template - Groq never called
