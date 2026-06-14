"""guardix + Google Gemini (google-genai SDK).

Needs GEMINI_API_KEY or GOOGLE_API_KEY (env or .env), and:
    pip install google-genai

Logs go to examples/logs/gemini.log - the console only shows responses.
"""

import os

from google import genai

from guardix import guard_client
from _common import ATTACK, BENIGN, load_dotenv, make_guardial, require_env, show

load_dotenv()
if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
    require_env("GEMINI_API_KEY")

MODEL = "gemini-2.0-flash"
client = guard_client(genai.Client(), guardial=make_guardial("gemini"))


def ask(label, prompt):
    r = client.models.generate_content(model=MODEL, contents=prompt)
    show(label, r.text)


ask("BENIGN", BENIGN)   # real Gemini answer
ask("ATTACK", ATTACK)   # block template - Gemini never called
