"""Asking a model a question about one output, and reading the answer.

A judge is two things and only the first is a model: a transport that can
complete a prompt, and a *task* that knows what to ask and how to read the
reply. Splitting them is what lets the panel, the cache, the cost accounting
and the health table be written once for every kind of field.
"""

from __future__ import annotations

from .base import (
    Judge,
    JudgeError,
    JudgeReply,
    JudgeRequest,
    JudgeTask,
    ParsedReply,
    Provider,
    ReplyOutcome,
)
from .prompts import ClaimSupportTask, GroundednessTask, LabelCorrectTask

__all__ = [
    "ClaimSupportTask",
    "GroundednessTask",
    "Judge",
    "JudgeError",
    "JudgeReply",
    "JudgeRequest",
    "JudgeTask",
    "LabelCorrectTask",
    "ParsedReply",
    "Provider",
    "ReplyOutcome",
]
