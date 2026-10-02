"""Grading labels against human answers: buckets, F1, direction, judges."""

from __future__ import annotations

import pytest

from llm_expectations.calibration.quality import brier, reliability
from llm_expectations.metrics.agreement import annotator_agreement
from llm_expectations.metrics.classification import (
    ConfusionPair,
    TreeBucket,
    bucket,
    classify,
    confusion_direction,
    judge_direction,
    primary_labels,
)
from llm_expectations.taxonomy import load_taxonomy
from llm_expectations.types import Label, Output, Status, Verdict

from .conftest import EXAMPLE

PF = "billing.payment_failed"
CD = "billing.card_declined"
RR = "billing.refund_request"
PR = "access.password_reset"


@pytest.fixture(scope="module")
def jtbd():
    return load_taxonomy(EXAMPLE / "taxonomy.yml")


class TestTreeBuckets:
    """Each bucket points at a different fix; one 'wrong' throws that away."""

    @pytest.mark.parametrize(
        "predicted, expected",
        [
            (PF, TreeBucket.EXACT),
            (CD, TreeBucket.RIGHT_PARENT),  # sibling — a boundary problem
            ("billing", TreeBucket.TOO_SHALLOW),  # hedging
            (PR, TreeBucket.WRONG),  # not reading the item
            (None, TreeBucket.ABSTAINED),
            ("abstain", TreeBucket.ABSTAINED),
            ("access.login_broken", TreeBucket.UNKNOWN),
        ],
    )
    def test_each_kind_of_wrong_lands_in_its_own_bucket(self, jtbd, predicted, expected):
        assert bucket(predicted, PF, jtbd) is expected

    def test_an_abstention_is_not_a_wrong_answer(self, jtbd):
        assert bucket(None, PF, jtbd) is not TreeBucket.WRONG

    def test_a_grandparent_is_still_too_shallow(self, jtbd):
        assert bucket("billing", CD, jtbd) is TreeBucket.TOO_SHALLOW


class TestClassify:
    def _run(self, jtbd, pairs):
        outputs = {f"s-{i}": Output(f"s-{i}", {"jtbd": p}) for i, (_, p) in enumerate(pairs)}
        labels = [
            Label(f"s-{i}", "jtbd", t, "ann-1") for i, (t, _) in enumerate(pairs)
        ]
        return classify("jtbd", outputs, labels, jtbd, label_floor=3)

    def test_a_perfect_run_scores_one(self, jtbd):
        scored = self._run(jtbd, [(PF, PF)] * 5 + [(CD, CD)] * 5)
        assert scored.accuracy == 1.0
        assert scored.macro_f1 == 1.0

    def test_accuracy_is_reported_beside_the_baseline_it_must_beat(self, jtbd):
        # Nine of ten are one label. Guessing it every time scores 90%.
        scored = self._run(jtbd, [(PF, PF)] * 9 + [(CD, PF)])
        assert scored.accuracy == pytest.approx(0.9)
        assert scored.majority_baseline == pytest.approx(0.9)
        assert scored.beats_majority is False

    def test_macro_f1_punishes_ignoring_a_rare_label(self, jtbd):
        # Accuracy barely notices; macro F1 does, which is why it is the
        # headline.
        scored = self._run(jtbd, [(PF, PF)] * 9 + [(CD, PF)])
        assert scored.accuracy > 0.85
        assert scored.macro_f1 < 0.6

    def test_macro_f1_averages_only_over_labels_people_actually_used(self, jtbd):
        # The taxonomy has seven leaves. Averaging over all of them would
        # measure the tree's size, not the model.
        scored = self._run(jtbd, [(PF, PF)] * 5)
        assert scored.macro_f1 == 1.0

    def test_per_label_recall_under_a_half_is_flagged(self, jtbd):
        scored = self._run(jtbd, [(CD, PF)] * 3 + [(CD, CD)] * 2 + [(PF, PF)] * 5)
        assert "billing.card_declined" in [s.label for s in scored.weak_labels]

    def test_a_label_with_too_few_rows_is_marked_rather_than_trusted(self, jtbd):
        outputs = {"s-1": Output("s-1", {"jtbd": PF})}
        labels = [Label("s-1", "jtbd", PF, "ann-1")]
        scored = classify("jtbd", outputs, labels, jtbd, label_floor=30)
        assert scored.scores[0].under_floor
        assert not scored.weak_labels  # under the floor, so not flagged as weak

    def test_the_confusion_matrix_records_what_was_predicted_instead(self, jtbd):
        scored = self._run(jtbd, [(PF, CD)] * 3)
        assert scored.confusion[0] == ConfusionPair(truth=PF, predicted=CD, n=3)

    def test_abstentions_are_bucketed_apart_from_predictions(self, jtbd):
        scored = self._run(jtbd, [(PF, None)] * 2 + [(PF, PF)] * 3)
        assert scored.buckets[TreeBucket.ABSTAINED] == 2
        # An abstention is not a prediction, so it cannot be a false positive.
        assert all(s.predicted <= 3 for s in scored.scores)

    def test_only_the_first_annotator_is_the_answer_key(self):
        labels = [Label("s-1", "jtbd", CD, "ann-2"), Label("s-1", "jtbd", PF, "ann-1")]
        assert primary_labels(labels, "jtbd") == {"s-1": PF}


class TestConfusionDirection:
    """Symmetric means fix the taxonomy; one-way means fix the prompt."""

    def test_the_designs_own_example_comes_out_as_written(self):
        pairs = [
            ConfusionPair("payment_failed", "card_declined", 23),
            ConfusionPair("card_declined", "payment_failed", 19),
            ConfusionPair("payment_failed", "refund", 31),
            ConfusionPair("refund", "payment_failed", 2),
        ]
        by_pair = {tuple(d["pair"]): d for d in confusion_direction(pairs)}
        assert by_pair[("card_declined", "payment_failed")]["shape"] == "symmetric"
        assert "taxonomy" in by_pair[("card_declined", "payment_failed")]["verdict"]
        assert by_pair[("payment_failed", "refund")]["shape"] == "asymmetric"
        assert "prompt" in by_pair[("payment_failed", "refund")]["verdict"]

    def test_a_pair_seen_once_is_refused_a_direction(self):
        # One-nil is not evidence of a one-way bias; it is a pair seen once.
        result = confusion_direction([ConfusionPair("a", "b", 1)])
        assert result[0]["shape"] == "too few to tell"

    def test_a_thin_pair_is_still_listed_so_a_boundary_can_be_watched(self):
        assert confusion_direction([ConfusionPair("a", "b", 2)])

    def test_each_boundary_appears_once_not_once_per_direction(self):
        pairs = [ConfusionPair("a", "b", 20), ConfusionPair("b", "a", 18)]
        assert len(confusion_direction(pairs)) == 1


class TestJudgeDirection:
    def _verdicts(self, rows, judge="judge-a"):
        return [
            Verdict(judge, "label_correct", item, "jtbd", status, 0.8, "r", {}, {})
            for item, status in rows
        ]

    def _data(self, truths):
        outputs = {i: Output(i, {"jtbd": p}) for i, (p, _) in truths.items()}
        labels = [Label(i, "jtbd", t, "ann-1") for i, (_, t) in truths.items()]
        return outputs, labels

    def _mixed(self, wrong=10, right=10):
        truths = {f"w-{i}": (CD, PF) for i in range(wrong)}
        truths.update({f"r-{i}": (PF, PF) for i in range(right)})
        return truths

    def test_a_lenient_judge_is_named_lenient(self):
        truths = self._mixed()
        outputs, labels = self._data(truths)
        # Waves through every wrong label, rejects nothing correct.
        verdicts = self._verdicts([(item, Status.PASS) for item in truths])
        row = judge_direction(verdicts, outputs, labels, ("jtbd",))[0]
        assert row.approves_wrong_rate == 1.0
        assert row.rejects_right_rate == 0.0
        assert row.leaning == "lenient"

    def test_a_strict_judge_is_named_strict(self):
        truths = self._mixed()
        outputs, labels = self._data(truths)
        verdicts = self._verdicts([(item, Status.FAIL) for item in truths])
        row = judge_direction(verdicts, outputs, labels, ("jtbd",))[0]
        assert row.rejects_right_rate == 1.0
        assert row.approves_wrong_rate == 0.0
        assert row.leaning == "strict"

    def test_a_balanced_judge_is_neither(self):
        truths = self._mixed()
        outputs, labels = self._data(truths)
        verdicts = self._verdicts(
            [(item, Status.PASS if i % 2 else Status.FAIL) for i, item in enumerate(truths)]
        )
        assert judge_direction(verdicts, outputs, labels, ("jtbd",))[0].leaning == "balanced"

    def test_one_sided_evidence_cannot_establish_a_leaning(self):
        # Every item is genuinely wrong, so there is no rejects-right rate to
        # compare against. The approves-wrong rate is still reported; calling
        # the judge lenient would assert something about behaviour that was
        # never observed.
        truths = {f"s-{i}": (CD, PF) for i in range(10)}
        outputs, labels = self._data(truths)
        verdicts = self._verdicts([(f"s-{i}", Status.PASS) for i in range(10)])
        row = judge_direction(verdicts, outputs, labels, ("jtbd",))[0]
        assert row.approves_wrong_rate == 1.0
        assert row.rejects_right_rate is None
        assert row.leaning == "unknown"

    def test_a_judge_that_cannot_beat_approving_everything_is_visible(self):
        # Nine right, one wrong, and the judge approves all ten.
        truths = {f"s-{i}": (PF, PF) for i in range(9)}
        truths["s-9"] = (CD, PF)
        outputs, labels = self._data(truths)
        verdicts = self._verdicts([(f"s-{i}", Status.PASS) for i in range(10)])
        row = judge_direction(verdicts, outputs, labels, ("jtbd",))[0]
        assert row.accuracy == pytest.approx(0.9)
        assert row.always_approve_accuracy == pytest.approx(0.9)
        assert not row.accuracy > row.always_approve_accuracy

    def test_unscored_verdicts_are_not_graded(self):
        truths = {"s-0": (PF, PF)}
        outputs, labels = self._data(truths)
        verdicts = self._verdicts([("s-0", Status.UNSCORED)])
        assert judge_direction(verdicts, outputs, labels, ("jtbd",)) == []

    def test_an_item_with_no_human_answer_is_not_graded(self):
        outputs = {"s-0": Output("s-0", {"jtbd": PF})}
        verdicts = self._verdicts([("s-0", Status.PASS)])
        assert judge_direction(verdicts, outputs, [], ("jtbd",)) == []


class TestConfidenceCalibration:
    def test_a_calibrated_judge_has_a_small_error(self):
        import random

        rng = random.Random(0)
        stated, happened = [], []
        for _ in range(4000):
            p = rng.choice([0.55, 0.65, 0.75, 0.85, 0.95])
            stated.append(p)
            happened.append(rng.random() < p)
        curve = reliability(stated, happened)
        assert curve.ece < 0.03
        assert not curve.broken
        assert curve.direction is None

    def test_a_judge_whose_point_nine_means_point_six_is_caught(self):
        import random

        rng = random.Random(1)
        stated = [0.9] * 2000
        happened = [rng.random() < 0.6 for _ in range(2000)]
        curve = reliability(stated, happened)
        assert curve.broken
        assert curve.ece == pytest.approx(0.3, abs=0.03)

    def test_the_direction_of_the_miscalibration_is_named(self):
        import random

        rng = random.Random(2)
        stated, happened = [], []
        for p in (0.6, 0.8, 0.95):
            for _ in range(500):
                stated.append(p)
                happened.append(rng.random() < p - 0.25)
        assert "overconfident" in reliability(stated, happened).direction

    def test_confidences_that_add_nothing_are_called_out(self):
        import random

        rng = random.Random(3)
        # The confidence is unrelated to whether the judge was right.
        stated = [rng.choice([0.6, 0.9]) for _ in range(2000)]
        happened = [rng.random() < 0.7 for _ in range(2000)]
        assert reliability(stated, happened).beats_base_rate is False

    def test_a_thin_bin_gets_no_number_of_its_own(self):
        curve = reliability([0.1] * 3 + [0.9] * 100, [False] * 3 + [True] * 100)
        thin = [b for b in curve.bins if b.n < 20]
        assert thin and not thin[0].reportable

    def test_brier_rewards_being_right_and_confident(self):
        assert brier([0.9, 0.9], [True, True]) < brier([0.6, 0.6], [True, True])

    def test_nothing_in_produces_nothing_out(self):
        curve = reliability([], [])
        assert curve.ece is None and curve.n == 0


class TestAnnotatorAgreement:
    """Two humans failing to separate two labels is a taxonomy bug."""

    def test_agreement_counts_only_doubly_labelled_items(self):
        labels = [
            Label("s-1", "jtbd", PF, "ann-1"),
            Label("s-1", "jtbd", PF, "ann-2"),
            Label("s-2", "jtbd", CD, "ann-1"),  # only one opinion
        ]
        result = annotator_agreement(labels, "jtbd")
        assert result.compared == 1
        assert result.agreement == 1.0

    def test_disagreement_concentrated_on_one_boundary_is_reported(self):
        labels = []
        for i in range(10):
            labels.append(Label(f"s-{i}", "jtbd", PF, "ann-1"))
            labels.append(Label(f"s-{i}", "jtbd", CD if i < 6 else PF, "ann-2"))
        result = annotator_agreement(labels, "jtbd")
        assert result.disagreements == 6
        assert result.pairs[0][:2] == (CD, PF)
        assert result.concentration == 1.0

    def test_a_single_annotator_produces_no_comparison(self):
        labels = [Label(f"s-{i}", "jtbd", PF, "ann-1") for i in range(50)]
        result = annotator_agreement(labels, "jtbd")
        assert result.compared == 0
        assert result.agreement is None

    def test_it_reads_only_the_field_it_was_asked_about(self):
        labels = [
            Label("s-1", "jtbd", PF, "ann-1"),
            Label("s-1", "jtbd", PF, "ann-2"),
            Label("s-1", "outcome", "resolved", "ann-1"),
            Label("s-1", "outcome", "pending", "ann-2"),
        ]
        assert annotator_agreement(labels, "jtbd").agreement == 1.0
        assert annotator_agreement(labels, "outcome").agreement == 0.0
