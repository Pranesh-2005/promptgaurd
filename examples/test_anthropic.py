"""guardix + Anthropic Claude. Needs ANTHROPIC_API_KEY (env or .env).

Logs go to examples/logs/anthropic.log - the console only shows responses.
"""

import anthropic

from guardix import guard_client
from _common import ATTACK, BENIGN, load_dotenv, make_guardial, require_env, show

load_dotenv()
require_env("ANTHROPIC_API_KEY")

MODEL = "claude-haiku-4-5-20251001"
client = guard_client(anthropic.Anthropic(), guardial=make_guardial("anthropic"))


def ask(label, prompt):
    r = client.messages.create(
        model=MODEL, max_tokens=200, messages=[{"role": "user", "content": prompt}]
    )
    show(label, r.content[0].text)


ask("BENIGN", BENIGN)   # real Claude answer
ask("ATTACK", ATTACK)   # block template - Anthropic never called
