"""The panel: what its votes mean, what they are worth, and what it will not do."""

from __future__ import annotations

import random

import pytest

from llm_expectations.judges.panel import majority, sample_items
from llm_expectations.metrics import effective_votes, fuzzy_pairs, leniency, pairwise_agreement
from llm_expectations.types import Output, Status, Verdict


def v(judge, status, *, item="s-1", field="jtbd", confidence=0.8, reason="r", instead=None,
      reply="answered"):
    return Verdict(
        judge, "label_correct", item, field, status, confidence, reason,
        {"instead": instead} if instead else {}, {"reply": reply, "model": "m"},
    )


class TestSampling:
    def test_the_same_sample_comes_back_every_time(self):
        items = [f"s-{i}" for i in range(500)]
        assert sample_items(items, 50) == sample_items(items, 50)

    def test_the_sample_does_not_move_when_rows_are_reordered(self):
        # A sample that drifted would turn every cached verdict into a miss
        # and make two runs' panel numbers incomparable for no reason.
        items = [f"s-{i}" for i in range(500)]
        shuffled = items[:]
        random.Random(1).shuffle(shuffled)
        assert sample_items(items, 50) == sample_items(shuffled, 50)

    def test_asking_for_more_than_exists_takes_everything(self):
        assert len(sample_items([f"s-{i}" for i in range(10)], 300)) == 10

    def test_adding_rows_keeps_most_of_the_old_sample(self):
        before = set(sample_items([f"s-{i}" for i in range(200)], 50))
        after = set(sample_items([f"s-{i}" for i in range(400)], 50))
        assert len(before & after) >= 15  # hash order, not position


class TestMajority:
    MEMBERS = ("a", "b", "c")

    def test_two_of_three_passes_and_the_split_stays_visible(self):
        result = majority(
            [v("a", Status.PASS), v("b", Status.FAIL, reason="renewal problem"),
             v("c", Status.PASS)],
            members=self.MEMBERS,
        )
        assert result.status is Status.PASS
        assert result.agreement == "2 of 3"
        assert result.votes == {"a": "correct", "b": "incorrect", "c": "correct"}

    def test_the_dissent_is_quoted(self):
        # Hiding a 2-1 behind an average throws away the only interesting part.
        result = majority(
            [v("a", Status.PASS), v("b", Status.FAIL, reason="renewal problem"),
             v("c", Status.PASS)],
            members=self.MEMBERS,
        )
        assert result.dissent == {"b": "renewal problem"}

    def test_a_tie_is_the_panel_failing_to_decide_not_a_pass(self):
        result = majority([v("a", Status.PASS), v("b", Status.FAIL)], members=("a", "b"))
        assert result.status is Status.UNSCORED
        assert "tied" in result.agreement

    def test_cannot_decide_is_not_half_a_rejection(self):
        # Counting it as one would manufacture disagreement out of an honest
        # abstention.
        result = majority(
            [v("a", Status.PASS), v("b", Status.UNSCORED, reply="cannot_decide"),
             v("c", Status.PASS)],
            members=self.MEMBERS,
        )
        assert result.status is Status.PASS
        assert result.agreement == "2 of 2"
        assert result.votes["b"] == "cannot_decide"

    def test_nobody_deciding_is_unscored(self):
        result = majority(
            [v("a", Status.UNSCORED, reply="cannot_decide"),
             v("b", Status.UNSCORED, reply="unparseable")],
            members=("a", "b"),
        )
        assert result.status is Status.UNSCORED
        assert result.agreement == "nobody decided"

    def test_an_excluded_member_gets_no_vote(self):
        result = majority(
            [v("a", Status.PASS), v("b", Status.FAIL), v("c", Status.FAIL)],
            members=("a", "c"),  # b dropped for being a rubber stamp
        )
        assert "b" not in result.votes
        assert result.status is Status.UNSCORED  # 1-1 among those who count

    def test_suggestions_ride_along_on_rejections(self):
        result = majority(
            [v("a", Status.FAIL, instead="billing.card_declined"), v("b", Status.FAIL)],
            members=("a", "b"),
        )
        assert result.instead == {"a": "billing.card_declined"}


class TestEffectiveVotes:
    def _panel(self, judges, n=60, seed=0, flip=0.0):
        rng = random.Random(seed)
        rows = []
        for i in range(n):
            truth = rng.random() < 0.7
            for judge in judges:
                wrong = rng.random() < flip
                rows.append(
                    v(judge, Status.PASS if truth != wrong else Status.FAIL, item=f"s-{i}")
                )
        return rows

    def test_identical_judges_are_worth_one_vote(self):
        report = effective_votes(self._panel("abc", flip=0.0), "abc")
        assert report.effective_votes == pytest.approx(1.0, abs=0.01)
        assert report.agreement == 1.0
        assert "paying for 3 opinions" in report.warning

    def test_independent_judges_are_worth_all_of_them(self):
        report = effective_votes(self._panel("abc", n=400, flip=0.5), "abc")
        assert report.effective_votes == pytest.approx(3.0, abs=0.3)
        assert report.warning is None

    def test_a_number_can_never_exceed_what_you_paid_for(self):
        # Judges disagreeing more than chance are still worth at most m.
        rows = [
            v(judge, Status.PASS if (i + k) % 2 else Status.FAIL, item=f"s-{i}")
            for i in range(40)
            for k, judge in enumerate("ab")
        ]
        report = effective_votes(rows, "ab")
        assert report.effective_votes is not None
        assert report.effective_votes <= 2.0

    def test_a_constant_judge_has_no_correlation_to_report(self):
        # Zero and one would each be a claim the data does not make.
        rows = [v("a", Status.PASS, item=f"s-{i}") for i in range(20)]
        rows += [
            v("b", Status.PASS if i % 2 else Status.FAIL, item=f"s-{i}") for i in range(20)
        ]
        report = effective_votes(rows, "ab")
        assert report.effective_votes is None
        assert "no independence to measure" in report.warning

    def test_a_panel_of_one_says_so(self):
        assert "panel of one" in effective_votes([v("a", Status.PASS)], ("a",)).warning

    def test_unscored_verdicts_are_not_compared(self):
        rows = [
            v("a", Status.PASS, item="s-1"),
            v("b", Status.UNSCORED, item="s-1", reply="cannot_decide"),
            v("a", Status.PASS, item="s-2"),
            v("b", Status.FAIL, item="s-2"),
        ]
        _, compared = pairwise_agreement(rows, "ab")
        assert compared == 1


class TestLeniency:
    def test_it_names_the_pushover(self):
        rows = []
        for i in range(10):
            rows.append(v("soft", Status.PASS, item=f"s-{i}"))
            rows.append(v("hard", Status.FAIL if i < 8 else Status.PASS, item=f"s-{i}"))
        entry = leniency(rows, ("soft", "hard"))[0]
        assert (entry.lenient, entry.strict) == ("soft", "hard")
        assert (entry.lenient_approved, entry.strict_approved) == (8, 0)

    def test_judges_that_never_split_are_not_listed(self):
        rows = [v(j, Status.PASS, item=f"s-{i}") for i in range(5) for j in "ab"]
        assert leniency(rows, "ab") == []


class TestFuzzyPairs:
    VOCAB = {"jtbd": frozenset({"billing.payment_failed", "billing.card_declined", "access.sso"})}

    def _split(self, item, assigned, instead):
        return [
            v("a", Status.FAIL, item=item, instead=instead),
            v("b", Status.PASS, item=item),
        ]

    def _outputs(self, mapping):
        return {k: Output(k, {"jtbd": val}) for k, val in mapping.items()}

    def test_the_pair_comes_from_the_dissenter_not_the_taxonomy(self):
        rows = self._split("s-1", "billing.payment_failed", "billing.card_declined")
        report = fuzzy_pairs(
            rows, "ab", self._outputs({"s-1": "billing.payment_failed"}), vocabularies=self.VOCAB
        )
        assert report.splits == 1
        assert report.pairs[0].left == "billing.card_declined"
        assert report.pairs[0].right == "billing.payment_failed"

    def test_a_split_with_no_suggestion_is_unattributed_not_guessed(self):
        rows = [v("a", Status.FAIL, item="s-1"), v("b", Status.PASS, item="s-1")]
        report = fuzzy_pairs(
            rows, "ab", self._outputs({"s-1": "billing.payment_failed"}), vocabularies=self.VOCAB
        )
        assert report.pairs == ()
        assert report.unattributed == 1

    def test_both_directions_are_one_boundary(self):
        # Counting a↔b apart from b↔a would split the evidence and bury it.
        rows = self._split("s-1", "billing.payment_failed", "billing.card_declined")
        rows += self._split("s-2", "billing.card_declined", "billing.payment_failed")
        outputs = self._outputs(
            {"s-1": "billing.payment_failed", "s-2": "billing.card_declined"}
        )
        report = fuzzy_pairs(rows, "ab", outputs, vocabularies=self.VOCAB)
        assert len(report.pairs) == 1
        assert report.pairs[0].splits == 2

    def test_a_label_the_field_does_not_define_is_discarded(self):
        # A judge naming a label from another taxonomy is a health problem.
        # Promoting it to a boundary would send someone to rewrite two
        # definitions over a pair that was never real.
        rows = self._split("s-1", "billing.payment_failed", "resolved")
        report = fuzzy_pairs(
            rows, "ab", self._outputs({"s-1": "billing.payment_failed"}), vocabularies=self.VOCAB
        )
        assert report.pairs == ()
        assert report.unusable == 1

    def test_judges_that_agree_contribute_no_pair(self):
        rows = [v(j, Status.FAIL, item="s-1", instead="billing.card_declined") for j in "ab"]
        report = fuzzy_pairs(
            rows, "ab", self._outputs({"s-1": "billing.payment_failed"}), vocabularies=self.VOCAB
        )
        assert report.splits == 0


class TestACollapsedPanel:
    """Exclusions can leave fewer than two voters. That is not a panel."""

    def test_one_surviving_judge_produces_no_panel_numbers(
        self, example, tmp_path, scripted
    ):
        from llm_expectations.judges.fake import FakeProvider, reply
        from llm_expectations.run import run

        strict = FakeProvider(model="strict", default=reply(False, 0.9, "wrong"))
        rubber = FakeProvider(model="rubber", default=reply(True, 0.98, "fine"))
        by_id = {"judge-a": strict, "judge-b": rubber, "judge-c": rubber}
        result = run(
            example, out=tmp_path / "out", provider_factory=lambda spec: by_id[spec.id]
        )
        panel = result.metrics["panel"]
        assert panel.get("collapsed")
        # One judge's verdict relabelled as a consensus would be the worst of
        # both: it reads as agreement and is a single opinion.
        assert "agreement" not in panel
        assert "effective_votes" not in panel
        assert not [f for f in result.findings if f.check == "label_correct_panel"]

    def test_the_report_says_why_rather_than_omitting_the_block(
        self, example, tmp_path
    ):
        from llm_expectations.judges.fake import FakeProvider, reply
        from llm_expectations.run import run

        strict = FakeProvider(model="strict", default=reply(False, 0.9, "wrong"))
        rubber = FakeProvider(model="rubber", default=reply(True, 0.98, "fine"))
        by_id = {"judge-a": strict, "judge-b": rubber, "judge-c": rubber}
        report = run(
            example, out=tmp_path / "out", provider_factory=lambda spec: by_id[spec.id]
        ).report
        assert "PANEL" in report
        assert "survived screening" in report
        assert "A panel needs two" in report
