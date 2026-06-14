"""promptguard + Azure OpenAI. Needs (env or examples/.env):

    AZURE_OPENAI_API_KEY=<your-key>
    AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
    AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
    AZURE_OPENAI_API_VERSION=2024-12-01-preview

Logs go to examples/logs/azure-openai.log - the console only shows responses.
"""

import os

from openai import AzureOpenAI

from promptguard import guard_client
from _common import ATTACK, BENIGN, load_dotenv, make_Guardial, require_env, show

load_dotenv()
require_env("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")

DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

azure = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=API_VERSION,
)
client = guard_client(azure, Guardial=make_Guardial("azure-openai"), provider="azure-openai")


def ask(label, prompt):
    r = client.chat.completions.create(
        model=DEPLOYMENT, messages=[{"role": "user", "content": prompt}]
    )
    show(label, r.choices[0].message.content)


ask("BENIGN", BENIGN)   # real Azure answer
ask("ATTACK", ATTACK)   # block template - Azure never called

# Injection buried in benign text -> still caught (sliding-window + sentence scan)
ask(
    "BURIED ATTACK",
    ("The weather today is lovely and I enjoy walking in the park with my dog. " * 30)
    + " " + ATTACK,
)
