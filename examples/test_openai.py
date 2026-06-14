"""guardix + OpenAI. Needs OPENAI_API_KEY (env or .env).

Logs go to examples/logs/openai.log - the console only shows responses.
"""

from openai import OpenAI

from guardix import guard_client
from _common import ATTACK, BENIGN, load_dotenv, make_guardial, require_env, show

load_dotenv()
require_env("OPENAI_API_KEY")

MODEL = "gpt-4o-mini"
client = guard_client(OpenAI(), guardial=make_guardial("openai"))


def ask(label, prompt):
    r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
    show(label, r.choices[0].message.content)


ask("BENIGN", BENIGN)   # real OpenAI answer
ask("ATTACK", ATTACK)   # block template - OpenAI never called
