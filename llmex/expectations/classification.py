"""Expectations for label-valued extractions.

Both checks here are deterministic and free. That is the point: for a
classification task, a gold set answers "is this right?" exactly, and no judge
is needed to establish the baseline a judge would have to beat.

The distribution check exists because accuracy alone will not tell you that a
classifier has quietly collapsed onto its majority label.
"""

from __future__ import annotations

from typing import Any

from ..batch import Batch, Context
from ..calibration import agreement_rate, cohens_kappa
from ..classification import normalise_label
from ..defaults import (
    CORPUS_DOC_ID,
    DEFAULT_MAX_DISTRIBUTION_SHIFT,
    DEFAULT_MAX_LABEL_SHARE,
    DEFAULT_MIN_DISTINCT_LABELS,
    GOLD_LABEL_KEY,
)
from ..expectation import SyncExpectation
from ..registry import EXPECTATIONS
from ..result import Result
from ..types import Evidence, Grain, Kind, Provenance, SkipReason
from .builtin import ExpectFieldTrustworthy


@EXPECTATIONS.plugin("expect_field_matches_gold")
class ExpectFieldMatchesGold(SyncExpectation):
    """Compare an extracted label against a human one carried on the record.

    Free, exact, and available only where somebody has done the labelling —
    which is why the interesting behaviour is what happens when they have not.
    A record without gold is reported unscored, never passed: an unlabelled
    record is not evidence of correctness, and letting it read green is how a
    gold set silently stops covering the corpus it is supposed to measure.
    """

    id = "expect_field_matches_gold"
    version = "1"
    kind = Kind.DETERMINISTIC
    grain = Grain.FIELD

    def check(self, batch: Batch, ctx: Context) -> list[Result]:
        targets = self.config.get("fields", ["*"])
        gold_key = self.config.get("gold_key", GOLD_LABEL_KEY)
        normalise = bool(self.config.get("normalise", True))
        out: list[Result] = []

        for rec in batch.for_fields(targets):
            prov = Provenance(self.id, self.version, prompt_version=rec.prompt_version)
            if gold_key not in rec.meta:
                out.append(
                    Result(
                        expectation_id=self.id,
                        grain=Grain.FIELD,
                        doc_id=rec.doc_id,
                        field_name=rec.field_name,
                        success=None,
                        severity=self.severity,
                        observed=rec.value,
                        evidence=Evidence(
                            "skipped",
                            {
                                "reason": SkipReason.NOT_APPLICABLE.value,
                                "detail": f"no '{gold_key}' on this record",
                            },
                        ),
                        skip_reason=SkipReason.NOT_APPLICABLE,
                        provenance=prov,
                    )
                )
                continue

            gold, got = rec.meta[gold_key], rec.value
            if normalise:
                ok = normalise_label(gold) == normalise_label(got)
            else:
                ok = str(gold) == str(got)
            out.append(
                Result(
                    expectation_id=self.id,
                    grain=Grain.FIELD,
                    doc_id=rec.doc_id,
                    field_name=rec.field_name,
                    success=ok,
                    score=1.0 if ok else 0.0,
                    severity=self.severity,
                    observed=rec.value,
                    evidence=Evidence(
                        "gold_match" if ok else "gold_mismatch",
                        {"expected": gold, "got": got, "normalised": normalise},
                    ),
                    provenance=prov,
                )
            )
        return out


@EXPECTATIONS.plugin("expect_label_distribution_stable")
class ExpectLabelDistributionStable(SyncExpectation):
    """Corpus-grain collapse and drift detector for categorical fields.

    Two failure modes it catches that accuracy cannot:

    - **Collapse.** A classifier that answers the majority label for everything
      still scores well on an unbalanced corpus. A share above `max_share`
      says the model stopped discriminating.
    - **Drift.** Compared against a `baseline` distribution, total variation
      distance above `max_shift` means the mix moved. Group by prompt_version
      in the sink and you learn which release moved it.
    """

    id = "expect_label_distribution_stable"
    version = "1"
    kind = Kind.STATISTICAL
    grain = Grain.CORPUS

    def check(self, batch: Batch, ctx: Context) -> list[Result]:
        targets = self.config.get("fields", ["*"])
        max_share = float(self.config.get("max_share", DEFAULT_MAX_LABEL_SHARE))
        min_distinct = int(self.config.get("min_distinct", DEFAULT_MIN_DISTINCT_LABELS))
        baseline: dict[str, float] = self.config.get("baseline") or {}
        max_shift = float(self.config.get("max_shift", DEFAULT_MAX_DISTRIBUTION_SHIFT))
        normalise = bool(self.config.get("normalise", True))

        by_field: dict[str, list[str]] = {}
        for rec in batch.for_fields(targets):
            if rec.value in (None, "", []):
                continue
            label = normalise_label(rec.value) if normalise else str(rec.value)
            by_field.setdefault(rec.field_name, []).append(label)

        out: list[Result] = []
        for name, labels in by_field.items():
            n = len(labels)
            counts: dict[str, int] = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            shares = {k: v / n for k, v in counts.items()}
            top, top_share = max(shares.items(), key=lambda kv: kv[1])

            failures: list[str] = []
            if top_share > max_share:
                failures.append(f"'{top}' holds {top_share:.1%} of the corpus")
            if len(counts) < min_distinct:
                failures.append(f"only {len(counts)} distinct label(s)")

            detail: dict[str, Any] = {
                "n": n,
                "distinct": len(counts),
                "shares": {k: round(v, 4) for k, v in sorted(shares.items())},
                "commonest": top,
                "commonest_share": round(top_share, 4),
                "max_share": max_share,
            }

            if baseline:
                # Total variation distance: half the L1 gap between the two
                # distributions, so it reads as "share of mass that moved".
                keys = set(shares) | set(baseline)
                tvd = sum(abs(shares.get(k, 0.0) - baseline.get(k, 0.0)) for k in keys) / 2
                detail["baseline_shift"] = round(tvd, 4)
                detail["max_shift"] = max_shift
                if tvd > max_shift:
                    failures.append(f"{tvd:.1%} of the distribution moved from baseline")

            if failures:
                detail["failures"] = failures

            out.append(
                Result(
                    expectation_id=self.id,
                    grain=Grain.CORPUS,
                    doc_id=CORPUS_DOC_ID,
                    field_name=name,
                    success=not failures,
                    score=1.0 - top_share,
                    severity=self.severity,
                    observed=top,
                    evidence=Evidence("label_distribution", detail),
                    provenance=Provenance(self.id, self.version),
                )
            )
        return out


@EXPECTATIONS.plugin("expect_extraction_label_trustworthy")
class ExpectExtractionLabelTrustworthy(ExpectFieldTrustworthy):
    """A judge-graded classification check.

    Everything that makes a model-based check safe is inherited unchanged:
    escalation from cheap-tier verdicts plus an audit stratum, budget
    reservation, the four skip paths, calibrated thresholds carrying their own
    provenance, and the plan-time refusal to block on an uncalibrated judge.

    What differs is only the question being asked, so only three things are
    overridden — the default strategy and the two evidence kinds. If you find
    yourself needing to override `validate()` here, the divergence belongs in
    the parent instead.
    """

    id = "expect_extraction_label_trustworthy"
    version = "1"
    default_strategy = "label_judge"
    field_evidence_kind = "label_verdict"
    document_evidence_kind = "label_verdict_doc"


@EXPECTATIONS.plugin("expect_judge_agrees_with_gold")
class ExpectJudgeAgreesWithGold(SyncExpectation):
    """Corpus-grain meta-evaluation: does the judge agree with the humans?

    This is the check that grades the grader, and it only means anything on a
    run where gold labels are present — an evaluation or calibration run, not a
    production one. With no gold it reports unscored rather than passing,
    because "we could not check the judge" and "the judge is fine" are
    different statements.

    Reports Cohen's kappa rather than raw agreement alone. On a skewed taxonomy
    a judge that rubber-stamps everything agrees with humans most of the time
    while carrying no information, and only kappa says so.

    It reads results the model tier already produced, so it costs nothing.
    """

    id = "expect_judge_agrees_with_gold"
    version = "1"
    kind = Kind.DERIVED  # must run after the judge it is grading
    grain = Grain.CORPUS

    def check(self, batch: Batch, ctx: Context) -> list[Result]:
        targets = self.config.get("fields", ["*"])
        gold_key = self.config.get("gold_key", GOLD_LABEL_KEY)
        min_kappa = float(self.config.get("min_kappa", 0.4))
        judged_by = self.config.get(
            "judged_by", "expect_extraction_label_trustworthy"
        )
        normalise = bool(self.config.get("normalise", True))

        # (doc_id, field) -> did the judge accept this label?
        verdicts = {
            (r.doc_id, r.field_name): r.success
            for r in ctx.prior_results
            if r.expectation_id == judged_by
            and r.grain is Grain.FIELD
            and r.success is not None
        }

        by_field: dict[str, tuple[list[str], list[str]]] = {}
        for rec in batch.for_fields(targets):
            accepted = verdicts.get((rec.doc_id, rec.field_name))
            if accepted is None or gold_key not in rec.meta:
                continue
            gold, got = rec.meta[gold_key], rec.value
            if normalise:
                human_right = normalise_label(gold) == normalise_label(got)
            else:
                human_right = str(gold) == str(got)
            judge_says, human_says = by_field.setdefault(rec.field_name, ([], []))
            judge_says.append("right" if accepted else "wrong")
            human_says.append("right" if human_right else "wrong")

        out: list[Result] = []
        for name in {r.field_name for r in batch.for_fields(targets)}:
            judge_says, human_says = by_field.get(name, ([], []))
            prov = Provenance(self.id, self.version)
            if not judge_says:
                out.append(
                    Result(
                        expectation_id=self.id,
                        grain=Grain.CORPUS,
                        doc_id=CORPUS_DOC_ID,
                        field_name=name,
                        success=None,
                        severity=self.severity,
                        evidence=Evidence(
                            "skipped",
                            {
                                "reason": SkipReason.NOT_APPLICABLE.value,
                                "detail": (
                                    f"no scored '{judged_by}' results with "
                                    f"'{gold_key}' to compare against"
                                ),
                            },
                        ),
                        skip_reason=SkipReason.NOT_APPLICABLE,
                        provenance=prov,
                    )
                )
                continue

            kappa = cohens_kappa(judge_says, human_says)
            raw = agreement_rate(judge_says, human_says)
            out.append(
                Result(
                    expectation_id=self.id,
                    grain=Grain.CORPUS,
                    doc_id=CORPUS_DOC_ID,
                    field_name=name,
                    success=kappa >= min_kappa,
                    score=kappa,
                    threshold=min_kappa,
                    severity=self.severity,
                    observed=round(kappa, 4),
                    evidence=Evidence(
                        "judge_agreement",
                        {
                            "n": len(judge_says),
                            "cohens_kappa": round(kappa, 4),
                            "raw_agreement": round(raw, 4),
                            "min_kappa": min_kappa,
                            "judged_by": judged_by,
                        },
                    ),
                    provenance=prov,
                )
            )
        return out
