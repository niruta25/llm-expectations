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


@pytest.fixture
def big_corpus(tmp_path):
    """A corpus large enough for a ranking claim, with a known error rate.

    Gate 2 refuses to run below its floors, and rightly — but that means
    nothing above M3 can be exercised on the thirteen-item example. This
    writes a real project to disk so the whole pipeline runs against it.

    The item text names the true label, so a scripted judge can be as right
    or as wrong as a test needs it to be.
    """
    import json
    import random

    LABELS = [
        "billing.payment_failed",
        "billing.card_declined",
        "billing.refund_request",
        "access.password_reset",
        "access.sso_issue",
        "product.bug_report",
        "product.feature_request",
    ]
    SIBLINGS = {
        "billing.payment_failed": "billing.card_declined",
        "billing.card_declined": "billing.payment_failed",
    }

    def build(n=400, error_rate=0.2, seed=0, second_annotator=0, sibling_bias=0.0):
        """`second_annotator` doubly-labels that many items; `sibling_bias`
        is the share of errors forced onto the confusable billing pair."""
        rng = random.Random(seed)
        root = tmp_path / f"corpus-{n}-{seed}"
        root.mkdir(parents=True, exist_ok=True)
        truth_by_item, items, outputs, labels = {}, [], [], []
        for i in range(n):
            truth = rng.choice(LABELS)
            wrong = rng.random() < error_rate
            if not wrong:
                assigned = truth
            elif rng.random() < sibling_bias and truth in SIBLINGS:
                assigned = SIBLINGS[truth]
            else:
                assigned = rng.choice([x for x in LABELS if x != truth])
            truth_by_item[f"s-{i}"] = (assigned, truth)
            items.append(
                {
                    "id": f"s-{i}",
                    "text": f"session {i} about {truth.replace('.', ' ')} "
                    + "detail " * rng.randint(3, 40),
                }
            )
            outputs.append({"item_id": f"s-{i}", "jtbd": assigned})
            labels.append(
                {"item_id": f"s-{i}", "field": "jtbd", "label": truth, "annotator": "ann-1"}
            )
            if i < second_annotator:
                # A second human who cannot separate the confusable pair
                # either — the strongest evidence a taxonomy is wrong.
                other = truth
                if truth in SIBLINGS and rng.random() < 0.5:
                    other = SIBLINGS[truth]
                labels.append(
                    {"item_id": f"s-{i}", "field": "jtbd", "label": other,
                     "annotator": "ann-2"}
                )

        for name, rows in (
            ("items.jsonl", items),
            ("outputs.jsonl", outputs),
            ("labels.jsonl", labels),
        ):
            (root / name).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        (root / "taxonomy.yml").write_text(
            (EXAMPLE / "taxonomy.yml").read_text(encoding="utf-8"), encoding="utf-8"
        )
        (root / "judges.yml").write_text(
            (EXAMPLE / "judges.yml").read_text(encoding="utf-8"), encoding="utf-8"
        )
        (root / "schema.yml").write_text(
            "item: s\nfields:\n  jtbd:\n    kind: assigned\n    taxonomy: jtbd@v4\n",
            encoding="utf-8",
        )
        (root / "run.yml").write_text(
            "run_id: big\nitems: items.jsonl\noutputs: outputs.jsonl\n"
            "labels: labels.jsonl\nschema: schema.yml\ntaxonomy: taxonomy.yml\n"
            "judges: judges.yml\n",
            encoding="utf-8",
        )
        return root, truth_by_item

    return build


@pytest.fixture
def oracle_judge():
    """A judge whose accuracy a test can dial, keyed on the item text."""
    import random

    from llm_expectations.judges.fake import FakeProvider, reply

    def build(truth_by_item, *, catches=0.8, false_alarms=0.05, seed=1, model="claude-sonnet-5"):
        rng = random.Random(seed)

        def answer(request):
            index = request.user.split("session ")[1].split(" ")[0]
            assigned, truth = truth_by_item[f"s-{index}"]
            rate = catches if assigned != truth else false_alarms
            # One confidence per verdict, whatever the truth. If the
            # confidence varied with the answer key, a judge whose *verdicts*
            # were pure noise would still rank perfectly, and a test for
            # "no better than chance" could never fail.
            if rng.random() < rate:
                return reply(False, 0.9, "mismatch")
            return reply(True, 0.85, "looks right")

        return FakeProvider(model=model, default=answer)

    return build


@pytest.fixture
def example(no_confirm):
    """The worked project, with the cost prompt stripped."""
    from llm_expectations.config import load_run

    return no_confirm(load_run(EXAMPLE / "run.yml"), max_usd=5.0)


@pytest.fixture
def provider(scripted):
    """A judge scripted against the fixture's planted defects."""
    from llm_expectations.judges.fake import claims, reply

    def unreadable_label_only(request):
        """One unreadable reply, on one question.

        Keyed on the prompt shape rather than the item, because s-13 is asked
        about twice — once for its label and once for its summary's claims.
        Garbage to both would double the parse-failure rate past the stop
        ceiling for a reason about the test rather than the judge; and an
        *assigned*-shaped reply to a claim question is itself unreadable, so
        the fallback has to answer in the shape it was asked.
        """
        if "ASSIGNED LABEL:" in request.user:
            return "not json at all"
        return claims(supported=["the text restates the item"])

    return scripted(
        rules=(
            ("fraud alert", reply(False, 0.81, "the issuer refused the card")),
            ("year up front", reply(False, 0.74, "billing is a parent, not a leaf")),
            ("reset loop", reply(False, 0.88, "access.login_broken is not permitted")),
            ("Okta", reply("cannot_decide", 0.4, "the item does not say enough")),
            ("blank panel", unreadable_label_only),
        ),
        default=reply(True, 0.92, "the label matches"),
    )


@pytest.fixture
def two_runs(tmp_path, big_corpus):
    """Two finished runs over one items file, from two prompt versions.

    The shape a comparison is actually for: same corpus, same labels, two
    prompts. Returns the two run directories.
    """
    import dataclasses
    import json
    import random

    from llm_expectations.config import Budget, load_run
    from llm_expectations.judges.fake import FakeProvider, reply
    from llm_expectations.run import run

    LABELS = [
        "billing.payment_failed",
        "billing.card_declined",
        "billing.refund_request",
        "access.password_reset",
        "access.sso_issue",
        "product.bug_report",
        "product.feature_request",
    ]

    def build(n=400, rates=(0.30, 0.15), prompts=("p7", "p8"), taxonomy_of_b=None):
        root, _ = big_corpus(n=n, error_rate=rates[0])
        truth = {
            json.loads(line)["item_id"]: json.loads(line)["label"]
            for line in (root / "labels.jsonl").read_text().splitlines()
        }
        directories = []
        for prompt, rate in zip(prompts, rates, strict=True):
            rng = random.Random(hash(prompt) % 997)
            rows = []
            for item_id, real in truth.items():
                wrong = rng.random() < rate
                rows.append(
                    {
                        "item_id": item_id,
                        "jtbd": rng.choice([x for x in LABELS if x != real]) if wrong else real,
                    }
                )
            outputs = root / f"outputs_{prompt}.jsonl"
            outputs.write_text(
                "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
            )
            run_file = root / f"run_{prompt}.yml"
            run_file.write_text(
                f"run_id: jtbd-{prompt}\nitems: items.jsonl\noutputs: outputs_{prompt}.jsonl\n"
                f"labels: labels.jsonl\nschema: schema.yml\ntaxonomy: taxonomy.yml\n"
                f"judges: judges.yml\nproduced_by:\n  model: claude-sonnet-5\n"
                f"  prompt_version: {prompt}\n",
                encoding="utf-8",
            )
            config = dataclasses.replace(load_run(run_file), budget=Budget(None, False))
            result = run(
                config,
                out=tmp_path / "out",
                provider_factory=lambda spec: FakeProvider(
                    model="m", default=reply(True, 0.9, "ok")
                ),
            )
            directories.append(result.directory)

        if taxonomy_of_b is not None:
            manifest = json.loads((directories[1] / "run.json").read_text())
            manifest["taxonomies"] = {taxonomy_of_b: "adifferenthash"}
            (directories[1] / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root, directories[0], directories[1]

    return build
