"""Built-in expectations.

Named after the Great Expectations convention so the vocabulary is learnable,
but the semantics differ where they must: model-based checks emit results at
*both* grains from a single pass, because field pass-rates flatter a system
badly and document pass-rates alone tell you nothing about where to look.
"""

from __future__ import annotations

import random
from difflib import SequenceMatcher
from typing import Any

from ..batch import Batch, Context, ExtractionRecord
from ..defaults import (
    CORPUS_DOC_ID,
    DEFAULT_AUDIT_RATE,
    DEFAULT_ENSEMBLE_CALLS,
    DEFAULT_MIN_RATIO,
    DEFAULT_PROVIDER_ALIAS,
    DEFAULT_SAMPLING_SEED,
    GROUNDING_WINDOW_DIVISOR,
    MIN_ESTIMATED_ESCALATION_RATE,
    THRESHOLD_SOURCE_MANUAL,
    UNCALIBRATED_THRESHOLD,
)
from ..expectation import Expectation, SyncExpectation
from ..registry import EXPECTATIONS
from ..result import Result
from ..strategies.base import ScorePayload
from ..types import (
    Capability,
    Evidence,
    Grain,
    Kind,
    Provenance,
    SkipReason,
)

_EMPTY: tuple[Any, ...] = (None, "", [])


# --------------------------------------------------------------------------
# Tier 1 — deterministic, 100% coverage, free
# --------------------------------------------------------------------------


@EXPECTATIONS.plugin("expect_field_grounded_in_source")
class ExpectFieldGroundedInSource(SyncExpectation):
    """The highest-value check in the whole framework, and it costs nothing.

    If a value cannot be located in the source text, the model invented it.
    Resolution order: null, exact substring, declared span, sliding-window fuzzy.
    Each branch gets a distinct evidence kind so a failure is actionable.
    """

    id = "expect_field_grounded_in_source"
    version = "1"
    kind = Kind.DETERMINISTIC
    grain = Grain.FIELD

    def check(self, batch: Batch, ctx: Context) -> list[Result]:
        targets = self.config.get("fields", ["*"])
        min_ratio = float(self.config.get("min_ratio", DEFAULT_MIN_RATIO))
        allow_null = bool(self.config.get("allow_null", True))
        out: list[Result] = []

        for rec in batch.for_fields(targets):
            source = batch.source(rec.doc_id).text
            value = rec.value

            if value in _EMPTY:
                out.append(
                    self._result(
                        rec, allow_null, Evidence("null_value"), 1.0 if allow_null else 0.0
                    )
                )
                continue

            needle = str(value)
            if needle in source:
                out.append(self._result(rec, True, Evidence("exact_match"), 1.0))
                continue

            if rec.span:
                start, end = rec.span
                quoted = source[start:end]
                ratio = SequenceMatcher(None, needle, quoted).ratio()
                ev = Evidence(
                    "span_match",
                    {"span": list(rec.span), "source_text": quoted, "ratio": round(ratio, 3)},
                )
                out.append(self._result(rec, ratio >= min_ratio, ev, ratio))
                continue

            best, at = self._best_window(needle, source)
            ev = Evidence(
                "fuzzy_match",
                {"ratio": round(best, 3), "closest_source_text": at, "min_ratio": min_ratio},
            )
            out.append(self._result(rec, best >= min_ratio, ev, best))

        return out

    @staticmethod
    def _best_window(needle: str, source: str) -> tuple[float, str]:
        """O(n*m) worst case. M6 should add an n-gram index prefilter."""
        n = len(needle)
        if n == 0 or not source:
            return 0.0, ""
        best, at = 0.0, ""
        step = max(1, n // GROUNDING_WINDOW_DIVISOR)
        for i in range(0, max(1, len(source) - n + 1), step):
            window = source[i : i + n]
            r = SequenceMatcher(None, needle, window).ratio()
            if r > best:
                best, at = r, window
            if best == 1.0:
                break
        return best, at

    def _result(
        self, rec: ExtractionRecord, ok: bool, ev: Evidence, score: float
    ) -> Result:
        return Result(
            expectation_id=self.id,
            grain=Grain.FIELD,
            doc_id=rec.doc_id,
            field_name=rec.field_name,
            success=bool(ok),
            score=score,
            severity=self.severity,
            observed=rec.value,
            evidence=ev,
            provenance=Provenance(self.id, self.version, prompt_version=rec.prompt_version),
        )


@EXPECTATIONS.plugin("expect_field_type")
class ExpectFieldType(SyncExpectation):
    """Schema conformance. Cheap, and almost always passes when the extractor
    uses structured-output mode — which is exactly why it is necessary but
    nowhere near sufficient. Keep it for the day someone swaps to an
    unconstrained decode path."""

    id = "expect_field_type"
    version = "1"
    kind = Kind.DETERMINISTIC
    grain = Grain.FIELD

    _PY: dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def check(self, batch: Batch, ctx: Context) -> list[Result]:
        types: dict[str, str] = self.config.get("types", {})
        nullable = set(self.config.get("nullable", []))
        out: list[Result] = []
        for rec in batch.for_fields(list(types) or ["*"]):
            expected = types.get(rec.field_name)
            if expected is None:
                continue
            if rec.value is None:
                ok = rec.field_name in nullable
                ev = Evidence("null_check", {"nullable": ok})
            else:
                py = self._PY.get(expected, object)
                ok = isinstance(rec.value, py)
                ev = Evidence(
                    "type_check",
                    {"expected": expected, "got": type(rec.value).__name__},
                )
            out.append(
                Result(
                    expectation_id=self.id,
                    grain=Grain.FIELD,
                    doc_id=rec.doc_id,
                    field_name=rec.field_name,
                    success=ok,
                    severity=self.severity,
                    observed=rec.value,
                    evidence=ev,
                    provenance=Provenance(self.id, self.version),
                )
            )
        return out


@EXPECTATIONS.plugin("expect_fields_to_satisfy")
class ExpectFieldsToSatisfy(SyncExpectation):
    """Document-grain cross-field rule. The expression sees a dict of the
    document's fields. This is where derived-value consistency lives — totals,
    date ordering, referential rules.

    Design guidance: fields requiring computation are not extraction fields.
    Pull a derived value out of the model's job and compute it here from
    extracted primitives. You get determinism, testability, and a smaller
    surface for the expensive tier.
    """

    id = "expect_fields_to_satisfy"
    version = "1"
    kind = Kind.DETERMINISTIC
    grain = Grain.DOCUMENT

    def check(self, batch: Batch, ctx: Context) -> list[Result]:
        expr = self.config["expression"]
        label = self.config.get("label", expr)
        out: list[Result] = []
        for doc_id, recs in batch.by_document().items():
            fields = {r.field_name: r.value for r in recs}
            try:
                # TODO(M7): replace with a restricted AST evaluator. `eval` with
                # empty builtins is adequate for a trusted config file and
                # inadequate for anything user-supplied. Hard blocker for any
                # multi-tenant deployment.
                ok = bool(eval(expr, {"__builtins__": {}}, dict(fields)))  # noqa: S307
                ev = Evidence("rule", {"rule": label, "fields": fields})
            except Exception as exc:  # noqa: BLE001 - a broken rule is a failure
                ok = False
                ev = Evidence(
                    "rule_error", {"rule": label, "error": str(exc), "fields": fields}
                )
            out.append(
                Result(
                    expectation_id=self.id,
                    grain=Grain.DOCUMENT,
                    doc_id=doc_id,
                    success=ok,
                    severity=self.severity,
                    observed=label,
                    evidence=ev,
                    provenance=Provenance(self.id, self.version),
                )
            )
        return out


# --------------------------------------------------------------------------
# Tier 2 — statistical, corpus grain
# --------------------------------------------------------------------------


@EXPECTATIONS.plugin("expect_field_null_rate_between")
class ExpectFieldNullRateBetween(SyncExpectation):
    """Silent-degradation detector. Group by prompt_version in the sink and a
    regression stops being 'quality dropped' and becomes 'quality dropped when
    we shipped prompt v7'."""

    id = "expect_field_null_rate_between"
    version = "1"
    kind = Kind.STATISTICAL
    grain = Grain.CORPUS

    def check(self, batch: Batch, ctx: Context) -> list[Result]:
        lo = float(self.config.get("min_rate", 0.0))
        hi = float(self.config.get("max_rate", 1.0))
        out: list[Result] = []
        for name in batch.field_names:
            recs = [r for r in batch.records if r.field_name == name]
            if not recs:
                continue
            rate = sum(1 for r in recs if r.value in _EMPTY) / len(recs)
            out.append(
                Result(
                    expectation_id=self.id,
                    grain=Grain.CORPUS,
                    doc_id=CORPUS_DOC_ID,
                    field_name=name,
                    success=lo <= rate <= hi,
                    score=rate,
                    severity=self.severity,
                    observed=round(rate, 4),
                    evidence=Evidence(
                        "null_rate",
                        {"n": len(recs), "rate": round(rate, 4), "bounds": [lo, hi]},
                    ),
                    provenance=Provenance(self.id, self.version),
                )
            )
        return out


# --------------------------------------------------------------------------
# Tier 3 — model based
# --------------------------------------------------------------------------


@EXPECTATIONS.plugin("expect_field_trustworthy")
class ExpectFieldTrustworthy(Expectation):
    """Routes documents to a verifier and emits results at both grains.

    Sampling policy: every document that already failed a cheap deterministic
    check is escalated, plus a random audit stratum of the ones that passed.
    The audit stratum is what tells you whether the cheap gates work at all.
    """

    id = "expect_field_trustworthy"
    version = "1"
    kind = Kind.MODEL_BASED
    grain = Grain.FIELD
    required_capabilities = frozenset({Capability.STRUCTURED_OUTPUT})

    # Subclass overrides. Everything below this line — escalation, budget
    # reservation, the four skip paths, threshold provenance — is the same
    # regardless of what is being verified, so a variant supplies these three
    # and inherits the rest rather than reimplementing it.
    field_evidence_kind: str = "trust_score"
    document_evidence_kind: str = "trust_score_doc"

    def estimate_calls(self, batch: Batch) -> int:
        n_docs = len(batch.doc_ids)
        rate = float(self.config.get("audit_rate", DEFAULT_AUDIT_RATE))
        per_doc = self.config.get("_strategy_calls", DEFAULT_ENSEMBLE_CALLS)
        # Suspect documents escalate on top of the audit stratum, so the real
        # rate exceeds audit_rate by an amount the planner cannot yet know.
        return int(n_docs * max(rate, MIN_ESTIMATED_ESCALATION_RATE) * per_doc)

    async def validate(self, batch: Batch, ctx: Context) -> list[Result]:
        provider = ctx.providers[self.config.get("provider", DEFAULT_PROVIDER_ALIAS)]
        strategy = ctx.strategies[self.config.get("strategy", self.default_strategy)]
        cal_id = self.config.get("calibration")
        calibration = ctx.calibrations.get(cal_id) if cal_id else None
        audit_rate = float(self.config.get("audit_rate", DEFAULT_AUDIT_RATE))
        # Seeded so the same batch produces the same audit sample across runs;
        # otherwise two runs are not comparable. Not a security decision.
        rng = random.Random(self.config.get("seed", DEFAULT_SAMPLING_SEED))  # noqa: S311

        fallback = self.config.get("threshold")
        prov = Provenance(
            self.id,
            self.version,
            provider_id=provider.id,
            model_version=provider.model_version,
            strategy_id=strategy.id,
        )

        # Which fields this check is responsible for. Honouring it is not
        # cosmetic: a verifier grades — and bills for — every field handed to
        # it, so a config naming two fields on a twelve-field schema must not
        # quietly pay for the other ten.
        targets = self.config.get("fields", ["*"])
        wanted = None if targets == ["*"] else set(targets)

        out: list[Result] = []
        for doc_id, all_recs in batch.by_document().items():
            recs = (
                all_recs if wanted is None else [r for r in all_recs if r.field_name in wanted]
            )
            if not recs:
                continue

            suspect = any(
                ctx.prior.get((doc_id, r.field_name), True) is False for r in recs
            )
            escalate = suspect or rng.random() < audit_rate

            if not escalate:
                out.extend(self._skipped(doc_id, recs, SkipReason.SAMPLED_OUT, prov))
                continue

            est = strategy.estimate_calls(len(recs))
            if ctx.budget is not None and not await ctx.budget.reserve(est):
                out.extend(self._skipped(doc_id, recs, SkipReason.BUDGET_EXHAUSTED, prov))
                continue

            # Carried through from the extraction, not obtained here: under
            # observe-only the framework never re-runs the generator.
            logprobs = {
                r.field_name: float(r.meta["logprob"]) for r in recs if "logprob" in r.meta
            }
            payload = ScorePayload(
                doc_id=doc_id,
                source_text=batch.source(doc_id).text,
                extraction={r.field_name: r.value for r in recs},
                schema=batch.schema,
                logprobs=logprobs or None,
            )
            try:
                scoreset = await strategy.score(payload, provider)
            except Exception as exc:  # noqa: BLE001 - a dead provider is a skip, not a crash
                out.extend(
                    self._skipped(doc_id, recs, SkipReason.PROVIDER_ERROR, prov, str(exc))
                )
                continue

            if ctx.budget is not None:
                await ctx.budget.record(scoreset.cost)

            for rec in recs:
                if rec.field_name not in scoreset.field_scores:
                    # The verifier returned nothing for this field. Substituting
                    # a middling score here would clear a middling threshold and
                    # read as a pass; the field simply was not scored.
                    out.append(self._unscored_field(doc_id, rec, prov))
                    continue
                score = scoreset.field_scores[rec.field_name]
                t, source = self._threshold_for(calibration, rec.field_name, fallback)
                out.append(
                    Result(
                        expectation_id=self.id,
                        grain=Grain.FIELD,
                        doc_id=doc_id,
                        field_name=rec.field_name,
                        success=score >= t,
                        score=score,
                        threshold=t,
                        threshold_source=source,
                        severity=self.severity,
                        observed=rec.value,
                        evidence=Evidence(
                            self.field_evidence_kind,
                            {
                                "explanation": scoreset.explanations.get(rec.field_name, ""),
                                "calls_used": scoreset.n_calls_used,
                                "calls_dropped": scoreset.n_calls_dropped,
                            },
                        ),
                        # Zero, deliberately. One call scored every field on
                        # this document, and the runner sums cost across rows:
                        # repeating it here would report the spend two or three
                        # times over. The document row below carries it.
                        provenance=prov,
                    )
                )

            doc_t, doc_source = self._threshold_for(calibration, None, fallback)
            fs = scoreset.field_scores
            out.append(
                Result(
                    expectation_id=self.id,
                    grain=Grain.DOCUMENT,
                    doc_id=doc_id,
                    success=scoreset.doc_score >= doc_t,
                    score=scoreset.doc_score,
                    threshold=doc_t,
                    threshold_source=doc_source,
                    severity=self.severity,
                    evidence=Evidence(
                        self.document_evidence_kind,
                        {
                            # Reported by the strategy, not assumed here: a
                            # strategy is free to roll fields up differently.
                            "aggregation": scoreset.aggregation,
                            "weakest_field": min(fs, key=lambda k: fs[k]) if fs else None,
                        },
                    ),
                    # The whole document's verification cost lands here, once.
                    cost=scoreset.cost,
                    provenance=prov,
                )
            )
        return out

    def _threshold_for(
        self, calibration: Any, field_name: str | None, fallback: float | None
    ) -> tuple[float, str]:
        """A threshold and its provenance, in that order of preference:
        the calibration, an explicitly configured number, then the documented
        uncalibrated default. The last two are stamped `manual` so a reviewer
        can see at a glance that nothing defended them."""
        if calibration is not None:
            t = calibration.threshold_for(field_name)
            if t is not None:
                return float(t), str(calibration.ref)
        if fallback is not None:
            return float(fallback), THRESHOLD_SOURCE_MANUAL
        return UNCALIBRATED_THRESHOLD, THRESHOLD_SOURCE_MANUAL

    def _unscored_field(
        self, doc_id: str, rec: ExtractionRecord, prov: Provenance
    ) -> Result:
        return Result(
            expectation_id=self.id,
            grain=Grain.FIELD,
            doc_id=doc_id,
            field_name=rec.field_name,
            success=None,
            severity=self.severity,
            observed=rec.value,
            evidence=Evidence(
                "skipped",
                {
                    "reason": SkipReason.NOT_APPLICABLE.value,
                    "detail": "the verifier returned no score for this field",
                },
            ),
            skip_reason=SkipReason.NOT_APPLICABLE,
            provenance=prov,
        )

    def _skipped(
        self,
        doc_id: str,
        recs: list[ExtractionRecord],
        reason: SkipReason,
        prov: Provenance,
        detail: str = "",
    ) -> list[Result]:
        """Unscored, never passed. Both grains, so coverage rot is visible."""
        ev = Evidence("skipped", {"reason": reason.value, "detail": detail})
        rows = [
            Result(
                expectation_id=self.id,
                grain=Grain.FIELD,
                doc_id=doc_id,
                field_name=r.field_name,
                success=None,
                severity=self.severity,
                evidence=ev,
                skip_reason=reason,
                provenance=prov,
            )
            for r in recs
        ]
        rows.append(
            Result(
                expectation_id=self.id,
                grain=Grain.DOCUMENT,
                doc_id=doc_id,
                success=None,
                severity=self.severity,
                evidence=ev,
                skip_reason=reason,
                provenance=prov,
            )
        )
        return rows
