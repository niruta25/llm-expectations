"""The judge seam: what is asked, how the answer is read, and what is never defaulted."""

from __future__ import annotations

import dataclasses
import inspect
import json

import httpx
import pytest

from llm_expectations.config import JudgeSpec
from llm_expectations.judges import Judge, JudgeError, LabelCorrectTask, ReplyOutcome
from llm_expectations.judges.base import JudgeRequest, JudgeTask
from llm_expectations.judges.fake import FakeProvider, reply
from llm_expectations.judges.providers import build_provider, with_retries
from llm_expectations.judges.screening import screen
from llm_expectations.taxonomy import load_taxonomy
from llm_expectations.types import Item, Output, Severity, Status, Verdict

from .conftest import EXAMPLE


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy(EXAMPLE / "taxonomy.yml")


@pytest.fixture
def judge():
    return Judge("judge-a", FakeProvider(), temperature=0.0, max_tokens=60)


ITEM = Item("s-1", "Her card ending 4471 was declined at renewal.")
OUTPUT = Output("s-1", {"jtbd": "billing.card_declined"})


class TestLabelLeak:
    """The guarantee is a signature, not a rule someone has to remember."""

    def test_no_judge_entry_point_accepts_a_label(self):
        for function in (JudgeTask.build, Judge.ask, Judge.build, LabelCorrectTask.build):
            annotations = inspect.signature(function).parameters
            assert not any("Label" in str(p.annotation) for p in annotations.values()), function
            assert "label" not in {name.lower() for name in annotations}, function

    def test_the_collector_has_nowhere_to_receive_one(self):
        from llm_expectations.run import collect

        parameters = inspect.signature(collect).parameters
        assert "labels" not in parameters
        assert not any("Label" in str(p.annotation) for p in parameters.values())


class TestPrompt:
    def test_the_definitions_and_boundaries_go_in(self, taxonomy):
        request = LabelCorrectTask().build(ITEM, OUTPUT, "jtbd", taxonomy)
        assert "The customer's bank or card issuer refused the card." in request.system
        assert "not this:" in request.system
        assert "ASSIGNED LABEL: billing.card_declined" in request.user
        assert ITEM.text in request.user

    def test_cannot_decide_is_offered_rather_than_forced(self, taxonomy):
        assert "cannot_decide" in LabelCorrectTask().build(ITEM, OUTPUT, "jtbd", taxonomy).system

    def test_require_leaf_decides_which_labels_are_offered(self, taxonomy):
        strict = LabelCorrectTask(require_leaf=True).build(ITEM, OUTPUT, "jtbd", taxonomy)
        loose = LabelCorrectTask(require_leaf=False).build(ITEM, OUTPUT, "jtbd", taxonomy)
        # A field that demands a leaf must not show the judge the parents, or
        # the judge will start approving them.
        assert "\n  billing —" not in strict.system
        assert "\n  billing —" in loose.system

    def test_a_field_with_no_taxonomy_is_refused_rather_than_asked_vaguely(self):
        with pytest.raises(ValueError, match="guessing at the same boundaries"):
            LabelCorrectTask().build(ITEM, OUTPUT, "jtbd", None)

    def test_the_fingerprint_moves_when_a_definition_does(self, taxonomy):
        before = LabelCorrectTask().build(ITEM, OUTPUT, "jtbd", taxonomy).fingerprint
        edited = dataclasses.replace(
            taxonomy,
            nodes={
                **taxonomy.nodes,
                "billing.card_declined": dataclasses.replace(
                    taxonomy.get("billing.card_declined"), definition="Something else entirely."
                ),
            },
        )
        after = LabelCorrectTask().build(ITEM, OUTPUT, "jtbd", edited).fingerprint
        assert before != after


class TestParsing:
    def test_a_verdict_and_its_confidence_are_read(self):
        parsed = LabelCorrectTask().parse(reply(False, 0.7, "reads as a renewal problem"))
        assert (parsed.status, parsed.raw_confidence) == (Status.FAIL, 0.7)
        assert parsed.outcome is ReplyOutcome.ANSWERED

    def test_cannot_decide_is_unscored_never_wrong(self):
        parsed = LabelCorrectTask().parse(reply("cannot_decide", 0.4))
        assert parsed.status is Status.UNSCORED
        assert parsed.outcome is ReplyOutcome.CANNOT_DECIDE

    @pytest.mark.parametrize(
        "text", ["", "sure thing", "{not json", '{"correct": "maybe"}', "[1, 2]"]
    )
    def test_an_unreadable_reply_is_unscored_and_counted_never_defaulted(self, text):
        # A coin-flip default is uncorrelated by construction, which quietly
        # changes every agreement number in the run.
        parsed = LabelCorrectTask().parse(text)
        assert parsed.status is Status.UNSCORED
        assert parsed.outcome is ReplyOutcome.UNPARSEABLE

    def test_prose_around_the_json_is_tolerated(self):
        parsed = LabelCorrectTask().parse('```json\n{"correct": true, "confidence": 0.8}\n```')
        assert parsed.status is Status.PASS

    @pytest.mark.parametrize("value", [None, "high", 1.5, -0.2, True])
    def test_a_confidence_that_is_not_a_probability_becomes_none_not_a_guess(self, value):
        body = json.dumps({"correct": True, "confidence": value})
        parsed = LabelCorrectTask().parse(body)
        assert parsed.raw_confidence is None
        # Still a usable verdict — it just cannot be ranked, and the triage row
        # says so rather than being handed a stand-in number.
        assert parsed.status is Status.PASS


class TestAsking:
    def test_a_verdict_carries_the_model_prompt_and_tokens(self, judge, taxonomy):
        _, verdict = judge.ask(LabelCorrectTask(), ITEM, OUTPUT, "jtbd", taxonomy)
        assert verdict.judge_id == "judge-a"
        assert verdict.metadata["model"] == "fake-instruct"
        assert verdict.metadata["input_tokens"] > 0
        assert verdict.metadata["cache_hit"] is False

    def test_a_provider_failure_is_unscored_and_says_so(self, taxonomy):
        judge = Judge(
            "judge-a", FakeProvider(default=FakeProvider.FAILS), temperature=0.0, max_tokens=60
        )
        _, verdict = judge.ask(LabelCorrectTask(), ITEM, OUTPUT, "jtbd", taxonomy)
        assert verdict.status is Status.UNSCORED
        assert verdict.metadata["reply"] == ReplyOutcome.ERROR.value
        assert "provider call failed" in verdict.reason

    def test_an_unreadable_reply_keeps_the_text_for_the_guardrail(self, taxonomy):
        judge = Judge("judge-a", FakeProvider(default="nope"), temperature=0.0, max_tokens=60)
        _, verdict = judge.ask(LabelCorrectTask(), ITEM, OUTPUT, "jtbd", taxonomy)
        assert verdict.metadata["raw_reply"] == "nope"


class TestProviders:
    def _client(self, handler):
        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_the_anthropic_shape_is_read(self, monkeypatch):
        monkeypatch.setenv("KEY", "secret")
        seen = {}

        def handler(request):
            seen["headers"] = dict(request.headers)
            seen["body"] = request.content
            return httpx.Response(
                200,
                json={
                    "model": "claude-sonnet-5",
                    "content": [{"type": "text", "text": reply(True, 0.9)}],
                    "usage": {"input_tokens": 120, "output_tokens": 22},
                },
            )

        provider = build_provider(
            JudgeSpec("judge-a", "anthropic", "claude-sonnet-5", api_key_env="KEY"),
            client=self._client(handler),
        )
        result = provider.complete(JudgeRequest("sys", "usr", 60, 0.0))
        assert (result.input_tokens, result.output_tokens) == (120, 22)
        assert seen["headers"]["x-api-key"] == "secret"
        assert b'"temperature":0.0' in seen["body"].replace(b", ", b",")

    def test_the_openai_shape_is_read(self, monkeypatch):
        monkeypatch.setenv("KEY", "secret")

        def handler(request):
            assert request.headers["authorization"] == "Bearer secret"
            return httpx.Response(
                200,
                json={
                    "model": "gpt-5-mini",
                    "choices": [{"message": {"content": reply(False, 0.6)}}],
                    "usage": {"prompt_tokens": 90, "completion_tokens": 15},
                },
            )

        provider = build_provider(
            JudgeSpec("judge-b", "openai", "gpt-5-mini", api_key_env="KEY"),
            client=self._client(handler),
        )
        assert provider.complete(JudgeRequest("sys", "usr", 60, 0.0)).input_tokens == 90

    def test_a_local_endpoint_is_used_verbatim(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(200, json={"choices": [{"message": {"content": reply(True)}}]})

        provider = build_provider(
            JudgeSpec("judge-c", "openai_compatible", "qwen", endpoint="http://gpu-01:1234/v1"),
            client=self._client(handler),
        )
        provider.complete(JudgeRequest("sys", "usr", 60, 0.0))
        assert seen["url"] == "http://gpu-01:1234/v1/chat/completions"

    def test_a_missing_key_names_the_variable_rather_than_returning_a_401(self, monkeypatch):
        monkeypatch.delenv("ABSENT_KEY", raising=False)
        provider = build_provider(
            JudgeSpec("judge-a", "anthropic", "claude-sonnet-5", api_key_env="ABSENT_KEY"),
            client=self._client(lambda r: httpx.Response(200, json={})),
        )
        with pytest.raises(JudgeError, match="ABSENT_KEY"):
            provider.complete(JudgeRequest("sys", "usr", 60, 0.0))


class TestRetries:
    def _provider(self, statuses):
        calls = {"n": 0}

        class Flaky:
            id = "judge-a"
            model = "m"

            def complete(self, request):
                status = statuses[min(calls["n"], len(statuses) - 1)]
                calls["n"] += 1
                if status is None:
                    return httpx.Response(200)  # unused; success path below
                raise JudgeError(status)

        return Flaky(), calls

    def test_a_rate_limit_is_retried(self):
        provider, calls = self._provider(["HTTP 429: slow down"])
        with pytest.raises(JudgeError, match="after 3 retries"):
            with_retries(provider, retries=3, sleep=lambda _: None).complete(
                JudgeRequest("s", "u", 60, 0.0)
            )
        assert calls["n"] == 4

    def test_a_bad_request_is_not_retried_because_it_will_fail_identically(self):
        provider, calls = self._provider(["HTTP 400: malformed"])
        with pytest.raises(JudgeError):
            with_retries(provider, retries=3, sleep=lambda _: None).complete(
                JudgeRequest("s", "u", 60, 0.0)
            )
        assert calls["n"] == 1


class TestScreening:
    def _verdicts(self, statuses, outcome="answered", judge="judge-a"):
        return [
            Verdict(
                judge, "label_correct", f"s-{i}-{outcome}", "jtbd", status, 0.9, "r", {},
                {"reply": outcome, "model": "m"},
            )
            for i, status in enumerate(statuses)
        ]

    def test_approval_rate_is_over_what_the_judge_scored(self, settings):
        # A judge that abstains half the time and approves the rest is a 100%
        # approver. Measured over all calls it would read as 50% and pass.
        verdicts = self._verdicts([Status.PASS] * 10) + self._verdicts(
            [Status.UNSCORED] * 10, outcome="cannot_decide"
        )
        health = screen(verdicts, settings)[0]
        assert health.approval_rate == 1.0
        assert health.cannot_decide_rate == 0.5

    def test_a_rubber_stamp_is_flagged(self, settings):
        health = screen(self._verdicts([Status.PASS] * 19 + [Status.FAIL]), settings)[0]
        assert any(s is Severity.WARN for s, _ in health.flags)
        assert not health.excluded_from_panel

    def test_a_judge_past_the_stop_band_is_excluded_but_its_verdicts_remain(self, settings):
        health = screen(self._verdicts([Status.PASS] * 100), settings)[0]
        assert health.excluded_from_panel
        assert any("still on disk" in message for _, message in health.flags)

    def test_unreadable_replies_cross_the_warning_line(self, settings):
        verdicts = self._verdicts([Status.PASS] * 95) + self._verdicts(
            [Status.UNSCORED] * 5, outcome="unparseable"
        )
        health = screen(verdicts, settings)[0]
        assert health.unreadable_rate == 0.05
        assert any("unreadable" in message for _, message in health.flags)

    def test_a_judge_that_scored_nothing_has_no_rate_and_is_stopped(self, settings):
        health = screen(self._verdicts([Status.UNSCORED] * 5, outcome="unparseable"), settings)[0]
        assert health.approval_rate is None
        assert health.excluded_from_panel


class TestClaimSupport:
    """The one defect a free check cannot reach."""

    ITEM = Item(
        "s-1",
        "Maya wrote in March 3. Her card ending 4471 was declined when the "
        "subscription auto-renewed. She gave us a different card and the $49 "
        "charge went through.",
    )

    def _task(self, style="descriptive"):
        from llm_expectations.judges.prompts import ClaimSupportTask

        return ClaimSupportTask(style=style)

    def test_the_designs_worked_example_produces_its_stated_answer(self):
        from llm_expectations.judges.fake import claims

        parsed = self._task().parse(
            claims(
                supported=["her card was declined"],
                unsupported=["we issued a refund of $49"],
            )
        )
        assert parsed.status is Status.FAIL
        assert parsed.detail["support_rate"] == 0.5
        assert parsed.detail["unsupported"] == ["we issued a refund of $49"]

    def test_the_claim_list_rides_back_so_a_reviewer_sees_the_sentence(self):
        from llm_expectations.judges.fake import claims

        parsed = self._task().parse(claims(unsupported=["we refunded $49"]))
        assert parsed.detail["claims"][0]["claim"] == "we refunded $49"

    def test_all_supported_passes(self):
        from llm_expectations.judges.fake import claims

        parsed = self._task().parse(claims(supported=["a", "b"]))
        assert parsed.status is Status.PASS
        assert parsed.detail["support_rate"] == 1.0

    def test_confidence_reads_in_the_same_direction_as_every_other_verdict(self):
        from llm_expectations.judges.fake import claims

        clean = self._task().parse(claims(supported=["a", "b", "c", "d"]))
        dirty = self._task().parse(claims(unsupported=["a", "b", "c", "d"]))
        # Higher means more sure of what it just said, both ways.
        assert clean.raw_confidence == 1.0
        assert dirty.raw_confidence == 1.0

    @pytest.mark.parametrize("text", ["", "sure", "{broken", '{"claims": []}', '{"claims": 3}'])
    def test_an_unreadable_reply_is_unscored_not_defaulted(self, text):
        parsed = self._task().parse(text)
        assert parsed.status is Status.UNSCORED
        assert parsed.outcome is ReplyOutcome.UNPARSEABLE

    def test_a_missing_note_rides_along_when_offered(self):
        from llm_expectations.judges.fake import claims

        parsed = self._task().parse(
            claims(supported=["a"], missing="the charge eventually went through")
        )
        assert "eventually went through" in parsed.detail["missing"]

    def test_a_descriptive_field_is_asked_to_split_into_claims(self):
        request = self._task("descriptive").build(
            self.ITEM, Output("s-1", {"summary": "x"}), "summary", None
        )
        assert "separate factual claims" in request.system

    def test_a_judgement_is_asked_once_and_told_not_to_split(self):
        request = self._task("judgement").build(
            self.ITEM, Output("s-1", {"verdict": "resolved first contact"}), "verdict", None
        )
        assert "Do not split it up" in request.system

    def test_a_proposal_is_not_grounded_in_the_item_at_all(self):
        # Asking whether the item *states* a next action would fail every
        # single one: a proposal is by definition not in the record yet.
        request = self._task("proposal").build(
            self.ITEM, Output("s-1", {"next": "confirm the new card is default"}), "next", None
        )
        assert "about the future" in request.system
        assert "follows" in request.system

    def test_the_budget_is_bigger_than_an_assigned_verdict_needs(self):
        from llm_expectations.judges.prompts import CLAIM_TOKENS

        assert CLAIM_TOKENS > LabelCorrectTask().max_tokens


class TestTheFakeAnswersEachTaskInItsOwnShape:
    def test_a_claim_question_gets_a_claim_reply_by_default(self):
        from llm_expectations.judges.prompts import ClaimSupportTask

        provider = FakeProvider(default=reply(True, 0.9, "fine"))
        request = ClaimSupportTask().build(
            Item("s-1", "the card was declined"), Output("s-1", {"summary": "declined"}),
            "summary", None,
        )
        # An assigned-shaped reply to a claim question is unreadable, and a
        # judge screened out for that is screened out for a fault in the test.
        parsed = ClaimSupportTask().parse(provider.complete(request).text)
        assert parsed.outcome is ReplyOutcome.ANSWERED

    def test_an_assigned_question_still_gets_an_assigned_reply(self, taxonomy):
        provider = FakeProvider(default=reply(False, 0.7, "wrong"))
        request = LabelCorrectTask().build(ITEM, OUTPUT, "jtbd", taxonomy)
        parsed = LabelCorrectTask().parse(provider.complete(request).text)
        assert parsed.status is Status.FAIL
