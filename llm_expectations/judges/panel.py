"""Several judges on a sample, and what their split is allowed to mean.

The panel measures. It answers how good the output is over a sample, which
label pairs the judges keep splitting on, and whether any judge is broken
(DESIGN.md §6).

Three things it deliberately does not do, and each absence is load-bearing:

**It does not route disagreements to review.** Disagreement does not rank
errors; published results put it near chance and below any single judge. It
ships as a *scored baseline* instead, so your own corpus settles it — measured
by Gate 2 and used by nothing. Selecting it stays a deliberate act.

**It does not auto-approve on unanimity.** Judges agree constantly and are
wrong on a lot of what they agree about. Consensus is not proof.

**It does not report one blended score.** Hiding a 2–1 split behind an average
throws away the only interesting part, so every finding carries the votes and
quotes the dissent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..types import Status, Verdict

__all__ = ["CHECK_ID", "PanelResult", "majority", "sample_items"]

CHECK_ID = "label_correct_panel"


def sample_items(item_ids: Sequence[str], size: int, *, seed: str = "panel") -> tuple[str, ...]:
    """Pick the panel's sample, the same way every time.

    Ordered by a hash of the id rather than by position, so the sample does
    not move when rows are added, reordered or re-exported. A sample that
    drifted between runs would turn every cached verdict into a miss and make
    two runs' panel numbers incomparable for no reason.
    """
    if size >= len(item_ids):
        return tuple(item_ids)
    keyed = sorted(item_ids, key=lambda i: hashlib.sha256(f"{seed}\x00{i}".encode()).hexdigest())
    return tuple(sorted(keyed[:size]))


@dataclass(frozen=True, slots=True)
class PanelResult:
    """One item and field, as the panel saw it."""

    item_id: str
    field: str
    votes: Mapping[str, str]
    status: Status
    agreement: str
    dissent: Mapping[str, str]
    instead: Mapping[str, str]

    @property
    def split(self) -> bool:
        """Did the voting judges disagree with each other?"""
        return len(set(self.votes.values()) - {"cannot_decide", "unreadable"}) > 1


def majority(verdicts: Sequence[Verdict], *, members: Sequence[str]) -> PanelResult:
    """Count the votes, and keep the split visible.

    Only judges that actually decided get a vote: a ``cannot_decide`` is not
    half a rejection, and counting it as one would manufacture disagreement
    out of an honest abstention. A tie is not a majority — it is the panel
    failing to decide, and that is unscored rather than a pass.
    """
    by_judge = {v.judge_id: v for v in verdicts}
    votes: dict[str, str] = {}
    reasons: dict[str, str] = {}
    instead: dict[str, str] = {}
    for judge_id in members:
        verdict = by_judge.get(judge_id)
        if verdict is None:
            continue
        if verdict.status is Status.PASS:
            votes[judge_id] = "correct"
        elif verdict.status is Status.FAIL:
            votes[judge_id] = "incorrect"
            suggestion = verdict.detail.get("instead")
            if isinstance(suggestion, str):
                instead[judge_id] = suggestion
        else:
            votes[judge_id] = (
                "cannot_decide"
                if verdict.metadata.get("reply") == "cannot_decide"
                else "unreadable"
            )
        reasons[judge_id] = verdict.reason

    deciding = {j: v for j, v in votes.items() if v in {"correct", "incorrect"}}
    first = next(iter(verdicts), None)
    correct = sum(1 for v in deciding.values() if v == "correct")
    incorrect = len(deciding) - correct

    if not deciding:
        status, agreement = Status.UNSCORED, "nobody decided"
    elif correct > incorrect:
        status, agreement = Status.PASS, f"{correct} of {len(deciding)}"
    elif incorrect > correct:
        status, agreement = Status.FAIL, f"{incorrect} of {len(deciding)}"
    else:
        # An even split is the panel failing to decide. Calling it a pass
        # would let a 1–1 read as agreement.
        status, agreement = Status.UNSCORED, f"tied {correct}–{incorrect}"

    losing = "incorrect" if status is Status.PASS else "correct"
    dissent = {j: reasons[j] for j, v in deciding.items() if v == losing}

    return PanelResult(
        item_id=first.item_id if first else "",
        field=first.field if first else "",
        votes=votes,
        status=status,
        agreement=agreement,
        dissent=dissent,
        instead=instead,
    )
