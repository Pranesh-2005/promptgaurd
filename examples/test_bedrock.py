"""promptguard + AWS Bedrock (boto3 Converse API).

Needs AWS credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or an AWS
profile) and optionally AWS_REGION (defaults to us-east-1), plus:
    pip install boto3

Bedrock's client shape has no adapter yet, so this shows the direct-engine
pattern that works with ANY provider: analyze first, call the API only when
allowed, and render the block template yourself when not.

Logs go to examples/logs/bedrock.log - the console only shows responses.
"""

import os

import boto3

from promptguard.responses import render_block_message
from _common import ATTACK, BENIGN, load_dotenv, make_Guardial, show

load_dotenv()

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")

bedrock = boto3.client("bedrock-runtime", region_name=REGION)
g = make_Guardial("bedrock")


def ask(label, prompt):
    decision = g.guard(prompt, provider="bedrock")
    if decision.decision == "BLOCK":
        # Render the same block template the adapters use, log the action.
        g.logger.log_block_action(decision.prompt_id, "bedrock", "mock_response", decision.reason)
        show(label, render_block_message(decision))
        return
    r = bedrock.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    show(label, r["output"]["message"]["content"][0]["text"])


ask("BENIGN", BENIGN)   # real Bedrock answer
ask("ATTACK", ATTACK)   # block template - Bedrock never called
