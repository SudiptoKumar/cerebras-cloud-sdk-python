#!/usr/bin/env -S rye run python
"""Chat completion with automatic failover across multiple Cerebras API keys.

Reads an ordered list of API keys from the environment:

    CEREBRAS_API_KEY     the primary key (required)
    CEREBRAS_API_KEY_2   optional backup key(s), comma-separated

If a call fails with an authentication, quota, or rate-limit status
(401, 402, 403, or 429), that key is retired for the rest of the run
and the same call is retried with the next key. Any other error is
raised unchanged. If every key fails, the last error is re-raised
wrapped in a RuntimeError so the caller can alert a human.

This is the pattern long-running bots use to survive a single key
hitting its quota mid-run: preserve two keys, fail over in order,
and never let one exhausted key take the whole job down.

Run it with:

    CEREBRAS_API_KEY=... CEREBRAS_API_KEY_2=... ./examples/api_key_failover.py
"""

import os
import sys
from typing import List

from cerebras.cloud.sdk import APIStatusError, Cerebras

# Statuses that mean "this key cannot serve the call": bad or revoked
# key (401/403), quota exhausted (402), or rate limited (429).
FAILOVER_STATUS_CODES = {401, 402, 403, 429}


def load_api_keys() -> List[str]:
    """Collect the configured keys, primary first, duplicates removed."""
    raw = [os.environ.get("CEREBRAS_API_KEY", "")]
    raw += os.environ.get("CEREBRAS_API_KEY_2", "").split(",")
    keys: List[str] = []
    for key in raw:
        key = key.strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def create_with_failover(keys: List[str], **kwargs):  # type: ignore[no-untyped-def]
    """Run one chat completion, failing over across `keys` in order."""
    last_error = None
    for key in keys:
        client = Cerebras(api_key=key)
        try:
            return client.chat.completions.create(**kwargs)
        except APIStatusError as err:
            if err.status_code not in FAILOVER_STATUS_CODES:
                raise
            last_error = err
            print(
                f"key ...{key[-4:]} failed with status {err.status_code}; "
                "trying the next key",
                file=sys.stderr,
            )
    raise RuntimeError("all Cerebras API keys failed") from last_error


def main() -> None:
    keys = load_api_keys()
    if not keys:
        sys.exit("Set CEREBRAS_API_KEY (and optionally CEREBRAS_API_KEY_2) first.")
    response = create_with_failover(
        keys,
        model="gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": "In one sentence, what is a wafer-scale processor?",
            }
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
