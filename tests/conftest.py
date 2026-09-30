"""Shared fixtures: the worked example, and a builder for throwaway projects."""

from __future__ import annotations

from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "jtbd"

MINIMAL_TAXONOMY = """
id: jtbd
version: 4
labels:
  billing:
    definition: "Anything about money moving, or failing to move."
    not_this: ["cannot reach the page at all"]
    children:
      payment_failed:
        definition: "A charge was attempted and did not go through."
        examples: ["the renewal never went through"]
        not_this: ["asking for money back"]
      refund_request:
        definition: "The customer wants money returned to them."
        examples: ["wants the duplicate charge reversed"]
        not_this: ["a charge that never succeeded"]
"""

MINIMAL_SCHEMA = """
item: support_session
fields:
  jtbd:
    kind: assigned
    taxonomy: jtbd@v4
  summary:
    kind: free_text
    must_agree_with: [jtbd]
"""

MINIMAL_JUDGES = """
judges:
  - id: judge-a
    provider: anthropic
    model: claude-sonnet-5
    api_key_env: ANTHROPIC_API_KEY
  - id: judge-b
    provider: openai
    model: gpt-5-mini
    api_key_env: OPENAI_API_KEY
panel:
  members: [judge-a, judge-b]
triage:
  judge: judge-a
"""

MINIMAL_RUN = """
run_id: test-run
items: items.jsonl
outputs: outputs.jsonl
schema: schema.yml
taxonomy: taxonomy.yml
judges: judges.yml
"""


@pytest.fixture
def project(tmp_path: Path):
    """Write a loadable project, then let a test overwrite one file of it."""

    def build(**overrides: str) -> Path:
        files = {
            "taxonomy.yml": MINIMAL_TAXONOMY,
            "schema.yml": MINIMAL_SCHEMA,
            "judges.yml": MINIMAL_JUDGES,
            "run.yml": MINIMAL_RUN,
            "items.jsonl": '{"id": "s-1", "text": "her card was declined"}\n',
            "outputs.jsonl": '{"item_id": "s-1", "jtbd": "billing.payment_failed",'
            ' "summary": "the card was declined"}\n',
        }
        files.update(overrides)
        for name, body in files.items():
            (tmp_path / name).write_text(body, encoding="utf-8")
        return tmp_path / "run.yml"

    return build


@pytest.fixture
def scripted():
    """A judge that answers from a script keyed on the item text it is shown."""
    from llm_expectations.judges.fake import FakeProvider, reply

    def build(rules=(), default=None, **kwargs):
        return FakeProvider(
            rules=tuple(rules),
            default=default if default is not None else reply(True, 0.9, "looks right"),
            **kwargs,
        )

    return build


@pytest.fixture
def no_confirm():
    """Strip the cost prompt from a config, the way ``--yes`` does."""
    import dataclasses

    from llm_expectations.config import Budget

    def build(config, max_usd=None):
        return dataclasses.replace(config, budget=Budget(max_usd=max_usd, confirm=False))

    return build


@pytest.fixture
def settings():
    """Built-in defaults, with nothing overridden."""
    from llm_expectations.config import settings_from_mapping

    return settings_from_mapping({})
