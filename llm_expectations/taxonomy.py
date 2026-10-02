"""The label tree: parsing, lookups, version hashing, and static health checks.

One file has four readers (DESIGN.md §9) — the judge prompt, the validity
check, tree scoring, and cross-field consistency — plus a fifth that matters
most and lives outside the code: the human annotator reads the same
definitions, word for word. If the judge and the human read different
definitions, judge-versus-human disagreement tells you nothing about either.

That is why ``definition`` and ``not_this`` are load-bearing here and not
documentation, and why a missing definition is an error rather than a note.
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .types import ABSTAIN, Severity

__all__ = [
    "HealthIssue",
    "Migration",
    "Taxonomy",
    "TaxonomyChangedError",
    "TaxonomyError",
    "TaxonomyNode",
    "TaxonomyRef",
    "check_recorded_hash",
    "load_migration",
    "load_taxonomy",
    "parse_ref",
]

SEPARATOR = "."
_NODE_KEYS = frozenset({"definition", "examples", "not_this", "children"})
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

#: Default for the near-duplicate definition check. Two labels whose
#: definitions are this similar are usually one label wearing two hats.
DEFINITION_SIMILARITY = 0.8


class TaxonomyError(ValueError):
    """The taxonomy file could not be read as a taxonomy."""


class TaxonomyChangedError(TaxonomyError):
    """A taxonomy's content moved without its version moving with it.

    The worst silent failure this library guards against: someone tightens a
    definition, nobody bumps the version, and a month of results quietly stop
    being comparable to each other.
    """


@dataclass(frozen=True, slots=True)
class TaxonomyRef:
    """A pinned reference to a taxonomy, as fields name it: ``jtbd@v4``."""

    id: str
    version: int

    def __str__(self) -> str:
        return f"{self.id}@v{self.version}"


def parse_ref(ref: str) -> TaxonomyRef:
    """Parse ``"jtbd@v4"``. The ``@v`` is required — an unpinned reference is
    how two runs end up compared across a definition change."""
    match = re.fullmatch(r"([A-Za-z0-9_-]+)@v(\d+)", ref.strip())
    if match is None:
        raise TaxonomyError(
            f"taxonomy reference {ref!r} is not of the form 'id@vN', e.g. 'jtbd@v4'. "
            "The version is required: without it, two runs can be compared across a "
            "definition change without anyone noticing."
        )
    return TaxonomyRef(id=match.group(1), version=int(match.group(2)))


@dataclass(frozen=True, slots=True)
class TaxonomyNode:
    """One label in the tree."""

    path: str
    name: str
    definition: str
    depth: int
    parent: str | None
    children: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    not_this: tuple[str, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass(frozen=True, slots=True)
class HealthIssue:
    """One problem found by reading the file alone, before spending anything."""

    check: str
    severity: Severity
    message: str
    paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Taxonomy:
    """A parsed, versioned label tree."""

    id: str
    version: int
    nodes: Mapping[str, TaxonomyNode]
    content_hash: str
    source: Path | None = None

    @property
    def ref(self) -> TaxonomyRef:
        return TaxonomyRef(self.id, self.version)

    def __contains__(self, path: object) -> bool:
        return isinstance(path, str) and path in self.nodes

    def __iter__(self) -> Iterator[TaxonomyNode]:
        return iter(self.nodes.values())

    def __len__(self) -> int:
        return len(self.nodes)

    def get(self, path: str) -> TaxonomyNode:
        try:
            return self.nodes[path]
        except KeyError:
            raise TaxonomyError(f"{path!r} is not a label in {self.ref}") from None

    def is_leaf(self, path: str) -> bool:
        return self.get(path).is_leaf

    def parent_of(self, path: str) -> str | None:
        return self.get(path).parent

    def ancestors(self, path: str) -> tuple[str, ...]:
        """Root-first, excluding the label itself."""
        parts = path.split(SEPARATOR)
        return tuple(SEPARATOR.join(parts[: i + 1]) for i in range(len(parts) - 1))

    def siblings(self, path: str) -> tuple[str, ...]:
        """Labels sharing this one's parent, excluding it.

        Sibling pairs are where a model and a human both get confused, which is
        why the fuzzy-pair detector and the boundary-case check both start here.
        """
        parent = self.get(path).parent
        pool = self.nodes[parent].children if parent else self.roots()
        return tuple(p for p in pool if p != path)

    def roots(self) -> tuple[str, ...]:
        return tuple(n.path for n in self.nodes.values() if n.parent is None)

    def leaves(self) -> tuple[str, ...]:
        return tuple(n.path for n in self.nodes.values() if n.is_leaf)

    def health(
        self,
        *,
        require_examples: bool = True,
        definition_similarity: float = DEFINITION_SIMILARITY,
    ) -> list[HealthIssue]:
        """Everything wrong with the file that can be seen without data.

        Runs before you spend anything. ``require_examples`` controls whether a
        missing example is reported at all, not how loudly — it is a warning
        either way, because a label can be usable on its definition alone.
        """
        issues: list[HealthIssue] = []
        issues += self._missing_examples() if require_examples else []
        issues += self._duplicate_names()
        issues += self._near_duplicate_definitions(definition_similarity)
        issues += self._inconsistent_leaf_depth()
        issues += self._siblings_without_boundaries()
        return sorted(issues, key=lambda i: (list(Severity).index(i.severity), i.check, i.paths))

    # A missing definition is not reported here: it is rejected at parse time,
    # because every reader of this file — the judge prompt most of all — has
    # nothing to say about a label that does not define itself.

    def _missing_examples(self) -> list[HealthIssue]:
        bare = tuple(n.path for n in self.nodes.values() if not n.examples)
        if not bare:
            return []
        return [
            HealthIssue(
                "taxonomy_examples",
                Severity.WARN,
                f"{len(bare)} label(s) have no examples. A definition says what a label "
                "covers; an example is what a judge and an annotator pattern-match on.",
                bare,
            )
        ]

    def _duplicate_names(self) -> list[HealthIssue]:
        by_name: dict[str, list[str]] = {}
        for node in self.nodes.values():
            by_name.setdefault(node.name, []).append(node.path)
        return [
            HealthIssue(
                "taxonomy_duplicate_names",
                Severity.WARN,
                f"the name {name!r} appears on {len(paths)} branches. Two labels that read "
                "identically in a confusion matrix are two labels nobody can tell apart in "
                "a report.",
                tuple(sorted(paths)),
            )
            for name, paths in sorted(by_name.items())
            if len(paths) > 1
        ]

    def _near_duplicate_definitions(self, threshold: float) -> list[HealthIssue]:
        items = [(n.path, _normalise(n.definition)) for n in self.nodes.values()]
        issues = []
        for i, (path_a, def_a) in enumerate(items):
            for path_b, def_b in items[i + 1 :]:
                ratio = difflib.SequenceMatcher(None, def_a, def_b).ratio()
                if ratio >= threshold:
                    issues.append(
                        HealthIssue(
                            "taxonomy_similar_definitions",
                            Severity.WARN,
                            f"definitions are {ratio:.0%} similar. Two labels this close are "
                            "usually one label, and the judge will split on them at random.",
                            (path_a, path_b),
                        )
                    )
        return issues

    def _inconsistent_leaf_depth(self) -> list[HealthIssue]:
        depths = {self.nodes[p].depth for p in self.leaves()}
        if len(depths) <= 1:
            return []
        shallowest = min(depths)
        early = tuple(sorted(p for p in self.leaves() if self.nodes[p].depth == shallowest))
        return [
            HealthIssue(
                "taxonomy_leaf_depth",
                Severity.NOTE,
                f"leaves sit at depths {sorted(depths)}. Uneven depth is legitimate, but a "
                "branch that stops early by accident looks exactly like this, and "
                "`require_leaf` will accept the shallow one.",
                early,
            )
        ]

    def _siblings_without_boundaries(self) -> list[HealthIssue]:
        bare = tuple(
            sorted(
                n.path
                for n in self.nodes.values()
                if not n.not_this and self.siblings(n.path)
            )
        )
        if not bare:
            return []
        return [
            HealthIssue(
                "taxonomy_boundaries",
                Severity.NOTE,
                f"{len(bare)} label(s) with siblings carry no `not_this`. Definitions say "
                "what a label covers; boundary cases say where it stops, and that is "
                "exactly where a model and a human both get confused.",
                bare,
            )
        ]


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def load_taxonomy(path: str | Path) -> Taxonomy:
    """Read and validate a taxonomy file."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise TaxonomyError(f"no taxonomy file at {path}") from None
    except yaml.YAMLError as exc:
        raise TaxonomyError(f"{path} is not valid YAML: {exc}") from None
    return taxonomy_from_mapping(raw, source=path)


def taxonomy_from_mapping(raw: Any, *, source: Path | None = None) -> Taxonomy:
    """Build a taxonomy from an already-parsed mapping."""
    where = str(source) if source else "taxonomy"
    if not isinstance(raw, Mapping):
        raise TaxonomyError(f"{where} must be a mapping with 'id', 'version' and 'labels'")

    unknown = set(raw) - {"id", "version", "labels"}
    if unknown:
        raise TaxonomyError(f"{where}: unknown top-level key(s) {sorted(unknown)}")

    tax_id = raw.get("id")
    if not isinstance(tax_id, str) or not tax_id:
        raise TaxonomyError(f"{where}: 'id' is required and must be a string")

    version = raw.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise TaxonomyError(f"{where}: 'version' is required and must be an integer >= 1")

    labels = raw.get("labels")
    if not isinstance(labels, Mapping) or not labels:
        raise TaxonomyError(f"{where}: 'labels' is required and must be a non-empty mapping")

    nodes: dict[str, TaxonomyNode] = {}
    _walk(labels, parent=None, depth=0, nodes=nodes, where=where)

    return Taxonomy(
        id=tax_id,
        version=version,
        nodes=nodes,
        content_hash=content_hash(tax_id, version, nodes),
        source=source,
    )


def _walk(
    branch: Mapping[Any, Any],
    *,
    parent: str | None,
    depth: int,
    nodes: dict[str, TaxonomyNode],
    where: str,
) -> tuple[str, ...]:
    paths = []
    for name, body in branch.items():
        path = f"{parent}{SEPARATOR}{name}" if parent else str(name)
        _check_name(name, path, where)
        if not isinstance(body, Mapping):
            raise TaxonomyError(
                f"{where}: label {path!r} must be a mapping, got {type(body).__name__}"
            )

        unknown = set(body) - _NODE_KEYS
        if unknown:
            raise TaxonomyError(
                f"{where}: label {path!r} has unknown key(s) {sorted(unknown)}. "
                f"Permitted: {sorted(_NODE_KEYS)}"
            )

        definition = body.get("definition")
        if not isinstance(definition, str) or not definition.strip():
            raise TaxonomyError(
                f"{where}: label {path!r} has no definition. The definition is what goes "
                "into the judge prompt and in front of the annotator; a label without one "
                "is a label both of them have to guess at."
            )

        children_raw = body.get("children") or {}
        if not isinstance(children_raw, Mapping):
            raise TaxonomyError(f"{where}: 'children' of {path!r} must be a mapping")

        # Reserve the slot before recursing so children can be attached to it.
        nodes[path] = TaxonomyNode(
            path=path,
            name=str(name),
            definition=definition.strip(),
            depth=depth,
            parent=parent,
            examples=_strings(body.get("examples"), path, "examples", where),
            not_this=_strings(body.get("not_this"), path, "not_this", where),
        )
        children = _walk(children_raw, parent=path, depth=depth + 1, nodes=nodes, where=where)
        nodes[path] = dataclasses.replace(nodes[path], children=children)
        paths.append(path)
    return tuple(paths)


def _check_name(name: Any, path: str, where: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise TaxonomyError(
            f"{where}: label name {name!r} must be lowercase letters, digits and "
            f"underscores. The separator {SEPARATOR!r} builds paths and cannot appear "
            "in a name."
        )
    if name == ABSTAIN:
        raise TaxonomyError(
            f"{where}: {ABSTAIN!r} is reserved for 'the model declined to pick a label' "
            "and cannot be a label. A taxonomy that defines it makes abstention and a "
            "real answer indistinguishable in every distribution."
        )


def _strings(value: Any, path: str, key: str, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise TaxonomyError(f"{where}: '{key}' of {path!r} must be a list of strings")
    for entry in value:
        if not isinstance(entry, str):
            raise TaxonomyError(f"{where}: '{key}' of {path!r} contains a non-string entry")
    return tuple(str(entry).strip() for entry in value)


def content_hash(tax_id: str, version: int, nodes: Mapping[str, TaxonomyNode]) -> str:
    """Hash the taxonomy's *content*, not its bytes.

    Reformatting the YAML, reordering keys or editing a comment does not move
    this hash; changing a definition, an example, a boundary case or the shape
    of the tree does. "Any content change bumps the version" is the rule, so
    the hash has to mean content.
    """
    canonical = {
        "id": tax_id,
        "version": version,
        "labels": {
            path: {
                "definition": node.definition,
                "examples": sorted(node.examples),
                "not_this": sorted(node.not_this),
                "children": sorted(node.children),
            }
            for path, node in sorted(nodes.items())
        },
    }
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check_recorded_hash(taxonomy: Taxonomy, recorded: Mapping[str, str]) -> None:
    """Refuse to proceed if this version's content has moved since last time.

    ``recorded`` maps ``"jtbd@v4"`` to the hash seen when that reference was
    last used — read from ``index.jsonl`` once runs exist. A reference that has
    never been seen is fine; one whose content changed under a fixed version is
    not.
    """
    ref = str(taxonomy.ref)
    previous = recorded.get(ref)
    if previous is None or previous == taxonomy.content_hash:
        return
    raise TaxonomyChangedError(
        f"taxonomy {ref} has changed since it was last used.\n"
        f"  Recorded hash: {previous[:8]}...   Current: {taxonomy.content_hash[:8]}...\n"
        f"  Bump to v{taxonomy.version + 1}, or restore the file.\n"
        "  Comparing runs across a silent definition change is the failure this stops."
    )


@dataclass(frozen=True, slots=True)
class Migration:
    """How the labels of one taxonomy version map onto another's.

    Without one, two runs on different versions cannot be compared and the
    library refuses rather than lining up labels by name and hoping. A label
    that was renamed, or two that were merged, would otherwise read as a
    distribution that moved — the model's behaviour blamed for an edit to the
    taxonomy.

    ``None`` on the right-hand side means the label has no equivalent. Those
    items are excluded from any comparison and counted, because silently
    dropping them is how a distribution shift gets manufactured.
    """

    source: TaxonomyRef
    target: TaxonomyRef
    mapping: Mapping[str, str | None]
    path: Path | None = None

    def covers(self, taxonomy: Taxonomy) -> tuple[str, ...]:
        """Labels of the source version this mapping says nothing about."""
        return tuple(sorted(set(taxonomy.nodes) - set(self.mapping)))

    def translate(self, label: str) -> str | None:
        return self.mapping.get(label)

    @property
    def dropped(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, v in self.mapping.items() if v is None))

    @property
    def merged(self) -> dict[str, tuple[str, ...]]:
        """Targets that more than one source label now maps onto."""
        reverse: dict[str, list[str]] = {}
        for old, new in self.mapping.items():
            if new is not None:
                reverse.setdefault(new, []).append(old)
        return {k: tuple(sorted(v)) for k, v in sorted(reverse.items()) if len(v) > 1}


def load_migration(path: str | Path) -> Migration:
    """Read a migration file.

    The file names its own endpoints, so a mapping cannot be applied between
    the wrong pair of versions by accident::

        from: jtbd@v4
        to: jtbd@v5
        labels:
          billing.payment_failed: billing.charge_failed   # renamed
          billing.card_declined:  billing.charge_failed   # merged
          billing.refund_request: null                    # no equivalent
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise TaxonomyError(f"no migration file at {path}") from None
    except yaml.YAMLError as exc:
        raise TaxonomyError(f"{path} is not valid YAML: {exc}") from None

    if not isinstance(raw, Mapping):
        raise TaxonomyError(f"{path} must be a mapping with 'from', 'to' and 'labels'")
    unknown = set(raw) - {"from", "to", "labels"}
    if unknown:
        raise TaxonomyError(f"{path}: unknown key(s) {sorted(unknown)}")

    for key in ("from", "to"):
        if not isinstance(raw.get(key), str):
            raise TaxonomyError(
                f"{path}: '{key}' is required and names a version, e.g. 'jtbd@v4'. A "
                "mapping that does not say which versions it joins can be applied "
                "between the wrong pair without anyone noticing."
            )

    labels = raw.get("labels")
    if not isinstance(labels, Mapping) or not labels:
        raise TaxonomyError(f"{path}: 'labels' is required and must be a non-empty mapping")

    mapping: dict[str, str | None] = {}
    for old, new in labels.items():
        if not isinstance(old, str):
            raise TaxonomyError(f"{path}: label key {old!r} must be a string")
        if new is not None and not isinstance(new, str):
            raise TaxonomyError(
                f"{path}: {old!r} maps to {new!r}. Use a label name, or null for a label "
                "with no equivalent."
            )
        mapping[old] = new

    return Migration(
        source=parse_ref(str(raw["from"])),
        target=parse_ref(str(raw["to"])),
        mapping=mapping,
        path=path,
    )
