"""What fields exist, what kind each one is, and what can be set on it.

The kind is declared once per field and everything else follows from it
(DESIGN.md §"Three kinds of field"). Note what this module does *not* do: it
holds no checks and no judge prompts. Checks are registered against a kind in
``checks/`` so the runner walks a list rather than branching, and a lookup
keyed by kind is not a branch on kind.

``FIELD_SETTINGS`` is the registry of everything settable on a field. It exists
so that a mistyped threshold is an error instead of a line of YAML that does
nothing — which is the same class of silent failure the rest of the library is
about.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .taxonomy import TaxonomyRef, parse_ref

__all__ = [
    "FIELD_SETTINGS",
    "FieldKind",
    "FieldSpec",
    "Schema",
    "SchemaError",
    "Setting",
    "TextStyle",
    "ValueType",
    "schema_from_mapping",
    "validate_setting_value",
]


class SchemaError(ValueError):
    """The schema could not be read as a schema."""


class FieldKind(str, Enum):
    """The three kinds of field, and the one seam that is not built yet."""

    ASSIGNED = "assigned"  # a label chosen from a taxonomy
    FREE_TEXT = "free_text"  # a sentence written about the item
    COPIED = "copied"  # a value that is in the document


class ValueType(str, Enum):
    """What a copied value is, which decides how "is it in the document" is read.

    Exact string matching is the wrong test for most of these. ``$1,234.50``
    and ``1234.5`` are the same amount; ``March 3, 2026`` and ``2026-03-03``
    are the same date. A library that only matched characters would report
    the model as inventing values every time it tidied a format.
    """

    TEXT = "text"
    NUMBER = "number"
    DATE = "date"


class TextStyle(str, Enum):
    """How a free-text field is grounded, which decides the expensive check.

    ``descriptive`` splits into claims and checks each against the item.
    ``judgement`` is one conclusion, so it is asked as one question rather than
    split. ``proposal`` is about the future and is not grounded at all — the
    question becomes whether it follows from the facts.
    """

    DESCRIPTIVE = "descriptive"
    JUDGEMENT = "judgement"
    PROPOSAL = "proposal"


@dataclass(frozen=True, slots=True)
class Setting:
    """One thing that can be set, its default, and where it may be set.

    An empty ``kinds`` means the setting is project-wide and cannot appear in a
    field block — that is how ``config.GLOBAL_SETTINGS`` reuses this type.

    ``value_type`` is what makes a mistyped *value* an error and not just a
    mistyped key: ``max_label_share: "half"`` is caught before anything runs.
    ``PAIR`` means a two-element ascending band, like ``[0.01, 0.20]``.
    """

    name: str
    kinds: frozenset[FieldKind]
    default: Any
    value_type: type | str
    doc: str
    bounds: tuple[float, float] | None = None


#: A two-element ascending band rather than a scalar.
PAIR = "pair"


def _setting(
    name: str,
    kinds: set[FieldKind],
    default: Any,
    value_type: type | str,
    doc: str,
    bounds: tuple[float, float] | None = None,
) -> Setting:
    return Setting(name, frozenset(kinds), default, value_type, doc, bounds)


_ASSIGNED = {FieldKind.ASSIGNED}
_FREE_TEXT = {FieldKind.FREE_TEXT}
_COPIED = {FieldKind.COPIED}
_BOTH = {FieldKind.ASSIGNED, FieldKind.FREE_TEXT}
_ALL = {FieldKind.ASSIGNED, FieldKind.FREE_TEXT, FieldKind.COPIED}

#: Every threshold is a default you can change. Each one is printed in the
#: report next to the number it produced, together with which layer set it.
FIELD_SETTINGS: Mapping[str, Setting] = {
    s.name: s
    for s in (
        _setting("require_leaf", _ASSIGNED, True, bool, "a label must be a leaf, not a parent"),
        _setting(
            "allow_abstain", _ASSIGNED, True, bool, "the model may decline to pick a label"
        ),
        _setting(
            "max_label_share",
            _ASSIGNED,
            0.5,
            float,
            "warn when one label swallows more than this share of the batch",
            (0.0, 1.0),
        ),
        _setting(
            "abstain_rate",
            _ASSIGNED,
            (0.01, 0.20),
            PAIR,
            "warn outside this band — too high means the taxonomy has a gap, "
            "too low means the model is forcing a label onto unclear items",
            (0.0, 1.0),
        ),
        _setting(
            "max_drift_pp",
            _ASSIGNED,
            0.10,
            float,
            "warn when a label's share moves more than this against the previous run",
            (0.0, 1.0),
        ),
        _setting(
            "min_stability",
            _ASSIGNED,
            0.90,
            float,
            "warn below this when the field is regenerated and the model flips itself",
            (0.0, 1.0),
        ),
        _setting(
            "min_words",
            _FREE_TEXT,
            None,
            int,
            "shortest acceptable output; unset means the check reports unscored",
            (0, 100_000),
        ),
        _setting(
            "max_words",
            _FREE_TEXT,
            None,
            int,
            "longest acceptable output; unset means the check reports unscored",
            (1, 100_000),
        ),
        _setting(
            "min_specificity",
            _FREE_TEXT,
            0.90,
            float,
            "warn below this share of outputs carrying a rare detail from their item",
            (0.0, 1.0),
        ),
        _setting(
            "max_copy_ratio",
            _FREE_TEXT,
            0.5,
            float,
            "longest verbatim run as a share of the output; above this it is pasting",
            (0.0, 1.0),
        ),
        _setting(
            "min_claim_support",
            _FREE_TEXT,
            0.95,
            float,
            "the judge-backed one: share of claims the item actually supports",
            (0.0, 1.0),
        ),
        _setting(
            "require_verbatim",
            _COPIED,
            False,
            bool,
            "the value must appear character for character, not merely as the same "
            "amount or the same date written differently",
        ),
        _setting(
            "min_grounding",
            _COPIED,
            0.95,
            float,
            "warn below this share of judged values the document actually supports",
            (0.0, 1.0),
        ),
        _setting(
            "ambiguous_above",
            _COPIED,
            1,
            int,
            "more candidate values than this in one document and finding the value "
            "there proves little — the gate sends those rows to a judge",
            (1, 1000),
        ),
        _setting(
            "min_cross_field_agreement",
            _BOTH,
            0.90,
            float,
            "warn below this share of items whose fields agree with each other",
            (0.0, 1.0),
        ),
    )
}


def validate_setting_value(setting: Setting, value: Any, where: str) -> Any:
    """Check one override against its declared type and bounds.

    Used for both layers that can set something — ``settings.yml`` and a field
    block — so a bad value is rejected the same way wherever it was written.
    """
    name = setting.name
    if setting.value_type is PAIR:
        if isinstance(value, (str, Mapping)) or not isinstance(value, (list, tuple)):
            raise SchemaError(f"{where}: {name!r} must be a two-element band, e.g. [0.01, 0.20]")
        if len(value) != 2:
            raise SchemaError(f"{where}: {name!r} needs exactly two values, got {len(value)}")
        low, high = (_number(v, name, where) for v in value)
        if low > high:
            raise SchemaError(f"{where}: {name!r} band is descending: [{low}, {high}]")
        _in_bounds(setting, low, where)
        _in_bounds(setting, high, where)
        return (low, high)

    if setting.value_type is bool:
        if not isinstance(value, bool):
            raise SchemaError(f"{where}: {name!r} must be true or false, got {value!r}")
        return value

    number = _number(value, name, where)
    if setting.value_type is int:
        if isinstance(value, float) and not value.is_integer():
            raise SchemaError(f"{where}: {name!r} must be a whole number, got {value!r}")
        number = int(number)
    _in_bounds(setting, number, where)
    return number


def _number(value: Any, name: str, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{where}: {name!r} must be a number, got {value!r}")
    return float(value)


def _in_bounds(setting: Setting, value: float, where: str) -> None:
    if setting.bounds is None:
        return
    low, high = setting.bounds
    if not low <= value <= high:
        raise SchemaError(
            f"{where}: {setting.name!r} is {value}, outside the permitted range "
            f"[{low}, {high}]"
        )


_STRUCTURAL_KEYS = frozenset({"kind", "taxonomy", "style", "value_type", "must_agree_with"})


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One declared field: its kind, its wiring, and its own overrides.

    ``overrides`` holds only what this field's block set. It is deliberately
    not merged with the defaults here — merging loses the provenance that lets
    the report say where a threshold came from. ``config.Settings.resolve``
    does the merge and keeps the source.
    """

    name: str
    kind: FieldKind
    taxonomy: TaxonomyRef | None = None
    style: TextStyle | None = None
    value_type: ValueType | None = None
    must_agree_with: tuple[str, ...] = ()
    overrides: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.overrides is None:
            object.__setattr__(self, "overrides", {})


@dataclass(frozen=True, slots=True)
class Schema:
    """The fields under test, in declaration order."""

    item: str
    fields: Mapping[str, FieldSpec]

    def __getitem__(self, name: str) -> FieldSpec:
        try:
            return self.fields[name]
        except KeyError:
            raise SchemaError(f"no field {name!r} in the schema") from None

    def __contains__(self, name: object) -> bool:
        return name in self.fields

    def __len__(self) -> int:
        return len(self.fields)

    def of_kind(self, kind: FieldKind) -> tuple[FieldSpec, ...]:
        return tuple(f for f in self.fields.values() if f.kind is kind)

    def taxonomy_refs(self) -> tuple[TaxonomyRef, ...]:
        seen: dict[str, TaxonomyRef] = {}
        for spec in self.fields.values():
            if spec.taxonomy is not None:
                seen[str(spec.taxonomy)] = spec.taxonomy
        return tuple(seen.values())


def schema_from_mapping(raw: Any, *, where: str = "schema") -> Schema:
    """Build a schema from a parsed ``schema.yml``."""
    if not isinstance(raw, Mapping):
        raise SchemaError(f"{where} must be a mapping with 'item' and 'fields'")

    unknown = set(raw) - {"item", "fields"}
    if unknown:
        raise SchemaError(f"{where}: unknown top-level key(s) {sorted(unknown)}")

    item = raw.get("item")
    if not isinstance(item, str) or not item:
        raise SchemaError(f"{where}: 'item' is required and names the unit being processed")

    fields_raw = raw.get("fields")
    if not isinstance(fields_raw, Mapping) or not fields_raw:
        raise SchemaError(f"{where}: 'fields' is required and must be a non-empty mapping")

    fields = {str(name): _field(str(name), body, where) for name, body in fields_raw.items()}

    for spec in fields.values():
        for other in spec.must_agree_with:
            if other not in fields:
                raise SchemaError(
                    f"{where}: field {spec.name!r} must_agree_with {other!r}, which is not "
                    "a field in this schema"
                )
            if other == spec.name:
                raise SchemaError(f"{where}: field {spec.name!r} cannot agree with itself")

    return Schema(item=item, fields=fields)


def _field(name: str, body: Any, where: str) -> FieldSpec:
    if not isinstance(body, Mapping):
        raise SchemaError(f"{where}: field {name!r} must be a mapping")

    kind = _kind(name, body.get("kind"), where)

    permitted = _STRUCTURAL_KEYS | {s.name for s in FIELD_SETTINGS.values() if kind in s.kinds}
    for key in body:
        if key in _STRUCTURAL_KEYS:
            continue
        setting = FIELD_SETTINGS.get(str(key))
        if setting is None:
            raise SchemaError(
                f"{where}: field {name!r} sets unknown key {key!r}. A threshold that is not "
                f"read is a threshold that does nothing. Permitted here: {sorted(permitted)}"
            )
        if kind not in setting.kinds:
            applies = ", ".join(sorted(k.value for k in setting.kinds))
            raise SchemaError(
                f"{where}: field {name!r} is {kind.value} and cannot set {key!r}, "
                f"which applies to {applies} fields"
            )

    taxonomy = body.get("taxonomy")
    if kind is FieldKind.ASSIGNED:
        if not isinstance(taxonomy, str):
            raise SchemaError(
                f"{where}: assigned field {name!r} needs a pinned taxonomy, e.g. 'jtbd@v4'"
            )
        ref = parse_ref(taxonomy)
    else:
        if taxonomy is not None:
            raise SchemaError(f"{where}: field {name!r} is {kind.value} and has no taxonomy")
        ref = None

    style = _style(name, body.get("style"), kind, where)
    value_type = _value_type(name, body.get("value_type"), kind, where)
    agree = _must_agree_with(name, body.get("must_agree_with"), where)
    overrides = {
        str(k): validate_setting_value(FIELD_SETTINGS[str(k)], v, f"{where}: field {name!r}")
        for k, v in body.items()
        if k in FIELD_SETTINGS
    }

    return FieldSpec(
        name=name,
        kind=kind,
        taxonomy=ref,
        style=style,
        value_type=value_type,
        must_agree_with=agree,
        overrides=overrides,
    )


def _kind(name: str, value: Any, where: str) -> FieldKind:
    try:
        return FieldKind(value)
    except ValueError:
        permitted = ", ".join(k.value for k in FieldKind)
        raise SchemaError(
            f"{where}: field {name!r} has kind {value!r}. Permitted: {permitted}"
        ) from None


def _value_type(name: str, value: Any, kind: FieldKind, where: str) -> ValueType | None:
    if kind is not FieldKind.COPIED:
        if value is not None:
            raise SchemaError(f"{where}: field {name!r} is {kind.value} and has no value_type")
        return None
    if value is None:
        return ValueType.TEXT
    try:
        return ValueType(value)
    except ValueError:
        permitted = ", ".join(v.value for v in ValueType)
        raise SchemaError(
            f"{where}: field {name!r} has value_type {value!r}. Permitted: {permitted}"
        ) from None


def _style(name: str, value: Any, kind: FieldKind, where: str) -> TextStyle | None:
    if kind is not FieldKind.FREE_TEXT:
        if value is not None:
            raise SchemaError(f"{where}: field {name!r} is {kind.value} and has no style")
        return None
    if value is None:
        return TextStyle.DESCRIPTIVE
    try:
        return TextStyle(value)
    except ValueError:
        permitted = ", ".join(s.value for s in TextStyle)
        raise SchemaError(
            f"{where}: field {name!r} has style {value!r}. Permitted: {permitted}"
        ) from None


def _must_agree_with(name: str, value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise SchemaError(f"{where}: 'must_agree_with' of {name!r} must be a list of field names")
    return tuple(str(v) for v in value)
