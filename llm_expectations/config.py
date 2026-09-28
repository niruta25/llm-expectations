"""``run.yml`` and the three-layer settings merge.

Nothing has to be configured to start. Every default lives in a registry here
or in ``schema.py``, and resolving one returns the value *and where it came
from*, because a threshold printed without its source is a number nobody can
argue with (DESIGN.md §11).

    built-in defaults
         ↓  overridden by
    settings.yml            project-wide
         ↓  overridden by
    schema.yml, per field   the specific case

Judges are configured here rather than in ``judges/`` for one reason: a judge
is two things, and only the first is a model. ``judges.yml`` declares the
models and then wires them to the two *jobs* — a panel that measures and a
single judge that ranks — and both of those are config decisions that must
validate before anything is spent.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml

from .schema import (
    FIELD_SETTINGS,
    FieldSpec,
    Schema,
    SchemaError,
    Setting,
    schema_from_mapping,
    validate_setting_value,
)
from .taxonomy import Taxonomy, TaxonomyError, load_taxonomy

__all__ = [
    "GLOBAL_SETTINGS",
    "TRIAGE_STRATEGIES",
    "Budget",
    "ConfigError",
    "JudgeSpec",
    "PanelSpec",
    "ProducedBy",
    "Resolved",
    "RunConfig",
    "Settings",
    "TriageSpec",
    "load_run",
]

T = TypeVar("T")


class ConfigError(ValueError):
    """The configuration could not be read, or contradicts itself."""


def _global(name: str, default: Any, value_type: Any, doc: str, bounds: Any = None) -> Setting:
    return Setting(name, frozenset(), default, value_type, doc, bounds)


#: Project-wide settings. Everything here is tunable except one thing, which is
#: not in the registry at all: the label-leak assertion. That is not a
#: threshold, it is a bug check, and ``settings.yml`` refuses to set it.
GLOBAL_SETTINGS: Mapping[str, Setting] = {
    s.name: s
    for s in (
        _global(
            "bootstrap_resamples", 1000, int, "resamples behind every interval", (100, 100_000)
        ),
        _global(
            "judge_temperature",
            0.0,
            float,
            "a measuring instrument should not roll dice",
            (0.0, 2.0),
        ),
        _global("judge_max_tokens", 60, int, "verdict + confidence + one sentence", (1, 4000)),
        _global("judge_retries", 3, int, "retries per call, with backoff", (0, 10)),
        _global(
            "judge_approval_warn",
            (0.15, 0.85),
            "pair",
            "flag a judge approving outside this band — the rubber-stamp check",
            (0.0, 1.0),
        ),
        _global(
            "judge_approval_stop",
            (0.02, 0.98),
            "pair",
            "outside this, the judge is dropped from panel aggregates; its raw "
            "verdicts are still written to disk",
            (0.0, 1.0),
        ),
        _global(
            "parse_failure_warn",
            0.02,
            float,
            "above this share of unreadable replies you may be measuring your fallback",
            (0.0, 1.0),
        ),
        _global(
            "parse_failure_stop",
            0.05,
            float,
            "above this, judge-backed numbers are suppressed rather than reported",
            (0.0, 1.0),
        ),
        _global(
            "minority_class_min_items",
            30,
            int,
            "an AUC computed while ranking three negatives looks fine and is noise",
            (1, 100_000),
        ),
        _global("minority_class_min_share", 0.05, float, "the same floor, as a share", (0.0, 1.0)),
        _global(
            "effective_votes_warn",
            0.5,
            float,
            "flag when a panel of m judges is worth fewer than this fraction of m",
            (0.0, 1.0),
        ),
        _global("min_n_ranking", 200, int, "below this, no ranking claim is made", (1, 1_000_000)),
        _global(
            "min_n_distribution", 200, int, "below this, no drift claim is made", (1, 1_000_000)
        ),
        _global("min_n_per_label", 30, int, "below this, no per-label number", (1, 1_000_000)),
        _global("fuzzy_pair_report", 5, int, "how many confusable label pairs to name", (1, 100)),
        _global(
            "taxonomy_definition_similarity",
            0.8,
            float,
            "two definitions this alike are usually one label",
            (0.0, 1.0),
        ),
        _global("taxonomy_require_examples", True, bool, "report labels with no examples"),
        _global(
            "taxonomy_version_hash",
            True,
            bool,
            "refuse to run when a taxonomy's content moved without its version",
        ),
        _global("cost_confirm", True, bool, "ask before spending"),
    )
}

#: The ranking strategies ``triage.strategy`` may name. Four of them exist to
#: be beaten. ``triage/strategies.py`` registers against these names, so this
#: is the one list and a typo is caught before any calls are made.
TRIAGE_STRATEGIES: tuple[str, ...] = (
    "auto",
    "random",
    "output_length",
    "majority_label",
    "panel_disagreement",
    "raw_confidence",
    "calibrated_risk",
)

PROVIDERS = frozenset({"anthropic", "openai", "openai_compatible"})
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_BUDGETS: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)


@dataclass(frozen=True, slots=True)
class Resolved(Generic[T]):
    """A setting's value together with the layer that supplied it.

    ``source`` is what the report prints beside the number, and what
    ``Finding.threshold_from`` records.
    """

    value: T
    source: str

    def __str__(self) -> str:
        return f"{self.value} ({self.source})"


@dataclass(frozen=True, slots=True)
class Settings:
    """The project layer, and the merge that reads through to the defaults."""

    project: Mapping[str, Any]

    def resolve(self, key: str, field: FieldSpec | None = None) -> Resolved[Any]:
        """Value and provenance for one setting, in precedence order."""
        setting = FIELD_SETTINGS.get(key) or GLOBAL_SETTINGS.get(key)
        if setting is None:
            raise ConfigError(f"unknown setting {key!r}")
        if field is not None and key in field.overrides:
            return Resolved(field.overrides[key], f"schema.yml:{field.name}")
        if key in self.project:
            return Resolved(self.project[key], "settings.yml")
        return Resolved(setting.default, "default")

    def value(self, key: str, field: FieldSpec | None = None) -> Any:
        return self.resolve(key, field).value


def settings_from_mapping(raw: Any, *, where: str = "settings.yml") -> Settings:
    """Parse and validate ``settings.yml``.

    A field-scoped key is allowed here as a project-wide default — that is the
    middle layer's whole job. An unknown key is an error, for the same reason
    it is one in a field block.
    """
    if raw is None:
        return Settings(project={})
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where} must be a mapping of setting names to values")

    resolved: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key)
        if name == "label_leak_assertion":
            raise ConfigError(
                f"{where}: 'label_leak_assertion' cannot be set. It is not a threshold, it "
                "is a bug check — the code that builds triage signals does not accept "
                "labels, and the assertion confirms it. Everything else here is tunable."
            )
        setting = FIELD_SETTINGS.get(name) or GLOBAL_SETTINGS.get(name)
        if setting is None:
            raise ConfigError(f"{where}: unknown setting {name!r}")
        try:
            resolved[name] = validate_setting_value(setting, value, where)
        except SchemaError as exc:
            raise ConfigError(str(exc)) from None
    return Settings(project=resolved)


@dataclass(frozen=True, slots=True)
class JudgeSpec:
    """One declared judge. A model plus how to reach it — no job attached yet.

    ``api_key_env`` is the *name* of an environment variable, never a key. This
    file gets committed; the key never does, and a value that is not a valid
    variable name is rejected precisely because a pasted key is not one.
    """

    id: str
    provider: str
    model: str
    api_key_env: str | None = None
    endpoint: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class PanelSpec:
    """Several judges over a sample. Measures quality and finds fuzzy labels."""

    members: tuple[str, ...]
    sample: int = 300


@dataclass(frozen=True, slots=True)
class TriageSpec:
    """One judge over everything. Ranks what a human should open first."""

    judge: str
    scope: str | int = "all"
    strategy: str = "auto"
    budgets: tuple[float, ...] = DEFAULT_BUDGETS


@dataclass(frozen=True, slots=True)
class Judges:
    """The declared judges and the two jobs they are wired to."""

    judges: Mapping[str, JudgeSpec]
    panel: PanelSpec | None = None
    triage: TriageSpec | None = None


@dataclass(frozen=True, slots=True)
class ProducedBy:
    """What made the outputs. Stamped onto every result.

    Two runs from different prompt versions are not comparable, and the tool
    says so rather than drawing a trend line across a prompt change.
    """

    model: str
    prompt_version: str
    extra: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Budget:
    max_usd: float | None = None
    confirm: bool = True


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything one run needs, loaded and cross-checked."""

    run_id: str
    root: Path
    schema: Schema
    taxonomies: Mapping[str, Taxonomy]
    judges: Judges
    settings: Settings
    items: Path
    outputs: Path
    labels: Path | None = None
    produced_by: ProducedBy | None = None
    budget: Budget = Budget()

    def taxonomy_for(self, field_name: str) -> Taxonomy | None:
        spec = self.schema[field_name]
        return None if spec.taxonomy is None else self.taxonomies[spec.taxonomy.id]


def load_run(path: str | Path) -> RunConfig:
    """Load ``run.yml`` and everything it points at.

    Paths inside are relative to the file, so a project directory can be moved
    or checked out anywhere without editing it.
    """
    path = Path(path)
    raw = _yaml(path)
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path} must be a mapping")

    known = {
        "run_id", "items", "outputs", "labels", "schema", "taxonomy",
        "judges", "settings", "produced_by", "budget",
    }
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}. Permitted: {sorted(known)}")

    root = path.parent
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ConfigError(f"{path}: 'run_id' is required and names this run, e.g. 'jtbd-p8'")

    items = _required_path(raw, "items", root, path)
    outputs = _required_path(raw, "outputs", root, path)
    labels = _optional_path(raw, "labels", root, path)

    schema_path = _required_path(raw, "schema", root, path)
    try:
        schema = schema_from_mapping(_yaml(schema_path), where=str(schema_path))
    except SchemaError as exc:
        raise ConfigError(str(exc)) from None

    settings_path = _optional_path(raw, "settings", root, path)
    settings = settings_from_mapping(
        _yaml(settings_path) if settings_path else None,
        where=str(settings_path) if settings_path else "settings.yml",
    )

    taxonomies = _load_taxonomies(raw, root, path, schema)
    judges = _load_judges(raw, root, path)

    return RunConfig(
        run_id=run_id,
        root=root,
        schema=schema,
        taxonomies=taxonomies,
        judges=judges,
        settings=settings,
        items=items,
        outputs=outputs,
        labels=labels,
        produced_by=_produced_by(raw.get("produced_by"), path),
        budget=_budget(raw.get("budget"), path),
    )


def _yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"no file at {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from None


def _required_path(raw: Mapping[str, Any], key: str, root: Path, where: Path) -> Path:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where}: '{key}' is required and must be a path")
    return (root / value).resolve()


def _optional_path(raw: Mapping[str, Any], key: str, root: Path, where: Path) -> Path | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where}: '{key}' must be a path, or be left out")
    return (root / value).resolve()


def _load_taxonomies(
    raw: Mapping[str, Any], root: Path, where: Path, schema: Schema
) -> Mapping[str, Taxonomy]:
    """Load every taxonomy file and check the schema's pins resolve.

    A schema that asks for ``jtbd@v4`` against a file that now says version 5
    is the silent comparability failure in its other form, and it stops here.
    """
    declared = raw.get("taxonomy")
    paths: list[Path] = []
    if isinstance(declared, str):
        paths = [(root / declared).resolve()]
    elif isinstance(declared, (list, tuple)):
        paths = [(root / str(p)).resolve() for p in declared]
    elif declared is not None:
        raise ConfigError(f"{where}: 'taxonomy' must be a path or a list of paths")

    loaded: dict[str, Taxonomy] = {}
    for taxonomy_path in paths:
        try:
            taxonomy = load_taxonomy(taxonomy_path)
        except TaxonomyError as exc:
            raise ConfigError(str(exc)) from None
        if taxonomy.id in loaded:
            raise ConfigError(
                f"{where}: two taxonomy files both declare id {taxonomy.id!r} "
                f"({loaded[taxonomy.id].source} and {taxonomy_path})"
            )
        loaded[taxonomy.id] = taxonomy

    for ref in schema.taxonomy_refs():
        pinned = loaded.get(ref.id)
        if pinned is None:
            raise ConfigError(
                f"{where}: the schema pins {ref}, but no taxonomy file declares id "
                f"{ref.id!r}. Loaded: {sorted(loaded) or 'none'}"
            )
        if pinned.version != ref.version:
            raise ConfigError(
                f"{pinned.source}: the schema pins {ref}, but this file is version "
                f"{pinned.version}. Point the field at v{pinned.version}, or check out "
                "the version it was written against — silently running one against the "
                "other is how a month of results stops being comparable."
            )
    return loaded


def _load_judges(raw: Mapping[str, Any], root: Path, where: Path) -> Judges:
    judges_path = _optional_path(raw, "judges", root, where)
    if judges_path is None:
        return Judges(judges={})
    return judges_from_mapping(_yaml(judges_path), where=str(judges_path))


def judges_from_mapping(raw: Any, *, where: str = "judges.yml") -> Judges:
    """Parse ``judges.yml``: the models, then the two jobs."""
    if raw is None:
        return Judges(judges={})
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"{where} must be a mapping with 'judges', and optionally 'panel' and 'triage'"
        )
    unknown = set(raw) - {"judges", "panel", "triage"}
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}")

    declared = raw.get("judges") or []
    if not isinstance(declared, (list, tuple)):
        raise ConfigError(f"{where}: 'judges' must be a list")

    judges: dict[str, JudgeSpec] = {}
    for entry in declared:
        spec = _judge(entry, where)
        if spec.id in judges:
            raise ConfigError(f"{where}: two judges share the id {spec.id!r}")
        judges[spec.id] = spec

    return Judges(
        judges=judges,
        panel=_panel(raw.get("panel"), judges, where),
        triage=_triage(raw.get("triage"), judges, raw.get("panel") is not None, where),
    )


def _judge(entry: Any, where: str) -> JudgeSpec:
    if not isinstance(entry, Mapping):
        raise ConfigError(f"{where}: each judge must be a mapping")
    known = {"id", "provider", "model", "api_key_env", "endpoint", "temperature", "max_tokens"}
    unknown = set(entry) - known
    if unknown:
        raise ConfigError(f"{where}: judge has unknown key(s) {sorted(unknown)}")

    judge_id = entry.get("id")
    if not isinstance(judge_id, str) or not judge_id:
        raise ConfigError(f"{where}: every judge needs an 'id'")

    provider = entry.get("provider")
    if provider not in PROVIDERS:
        raise ConfigError(
            f"{where}: judge {judge_id!r} has provider {provider!r}. "
            f"Permitted: {sorted(PROVIDERS)}"
        )

    model = entry.get("model")
    if not isinstance(model, str) or not model:
        raise ConfigError(f"{where}: judge {judge_id!r} needs a 'model'")

    endpoint = entry.get("endpoint")
    if provider == "openai_compatible" and not endpoint:
        raise ConfigError(
            f"{where}: judge {judge_id!r} is openai_compatible and needs an 'endpoint', "
            "e.g. http://gpu-01:1234/v1"
        )

    api_key_env = entry.get("api_key_env")
    if api_key_env is not None:
        if not isinstance(api_key_env, str) or not _ENV_NAME.fullmatch(api_key_env):
            raise ConfigError(
                f"{where}: judge {judge_id!r} has api_key_env={api_key_env!r}, which is not "
                "a valid environment variable name. This field takes the NAME of the "
                "variable holding the key — 'ANTHROPIC_API_KEY' — because this file gets "
                "committed and the key never should be."
            )

    return JudgeSpec(
        id=judge_id,
        provider=str(provider),
        model=model,
        api_key_env=api_key_env,
        endpoint=str(endpoint) if endpoint else None,
        temperature=_opt_number(entry.get("temperature"), judge_id, "temperature", where),
        max_tokens=_opt_int(entry.get("max_tokens"), judge_id, "max_tokens", where),
    )


def _opt_number(value: Any, judge_id: str, key: str, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: judge {judge_id!r} has non-numeric {key}={value!r}")
    return float(value)


def _opt_int(value: Any, judge_id: str, key: str, where: str) -> int | None:
    number = _opt_number(value, judge_id, key, where)
    if number is None:
        return None
    if number != int(number):
        raise ConfigError(f"{where}: judge {judge_id!r} has non-integer {key}={value!r}")
    return int(number)


def _panel(raw: Any, judges: Mapping[str, JudgeSpec], where: str) -> PanelSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where}: 'panel' must be a mapping with 'members'")
    unknown = set(raw) - {"members", "sample"}
    if unknown:
        raise ConfigError(f"{where}: 'panel' has unknown key(s) {sorted(unknown)}")

    members = raw.get("members")
    if not isinstance(members, (list, tuple)) or not members:
        raise ConfigError(f"{where}: 'panel.members' must be a non-empty list of judge ids")
    members = tuple(str(m) for m in members)
    if len(set(members)) != len(members):
        raise ConfigError(f"{where}: 'panel.members' lists the same judge twice")
    for member in members:
        if member not in judges:
            raise ConfigError(
                f"{where}: panel member {member!r} is not a declared judge. "
                f"Declared: {sorted(judges) or 'none'}"
            )
    if len(members) < 2:
        raise ConfigError(
            f"{where}: a panel of one produces no agreement, no dissent and no effective "
            "vote count — every number the panel exists for is undefined. Use two or more, "
            "or wire this judge to triage instead."
        )

    sample = raw.get("sample", 300)
    if isinstance(sample, bool) or not isinstance(sample, int) or sample < 1:
        raise ConfigError(f"{where}: 'panel.sample' must be a positive whole number of items")
    return PanelSpec(members=members, sample=sample)


def _triage(
    raw: Any, judges: Mapping[str, JudgeSpec], has_panel: bool, where: str
) -> TriageSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where}: 'triage' must be a mapping with 'judge'")
    unknown = set(raw) - {"judge", "scope", "strategy", "budgets"}
    if unknown:
        raise ConfigError(f"{where}: 'triage' has unknown key(s) {sorted(unknown)}")

    judge = raw.get("judge")
    if not isinstance(judge, str) or judge not in judges:
        raise ConfigError(
            f"{where}: triage judge {judge!r} is not a declared judge. "
            f"Declared: {sorted(judges) or 'none'}"
        )

    scope = raw.get("scope", "all")
    if scope != "all" and (isinstance(scope, bool) or not isinstance(scope, int) or scope < 1):
        raise ConfigError(
            f"{where}: 'triage.scope' is 'all' or a sample size. A ranking only covers "
            "what you sampled, so the default is all."
        )

    strategy = raw.get("strategy", "auto")
    if strategy not in TRIAGE_STRATEGIES:
        raise ConfigError(
            f"{where}: unknown triage strategy {strategy!r}. "
            f"Permitted: {', '.join(TRIAGE_STRATEGIES)}"
        )
    if strategy == "panel_disagreement" and not has_panel:
        raise ConfigError(
            f"{where}: strategy 'panel_disagreement' needs a panel to disagree. "
            "Wire one, or pick another strategy."
        )

    return TriageSpec(
        judge=judge,
        scope=scope,
        strategy=str(strategy),
        budgets=_budgets(raw.get("budgets"), where),
    )


def _budgets(raw: Any, where: str) -> tuple[float, ...]:
    if raw is None:
        return DEFAULT_BUDGETS
    if isinstance(raw, (str, Mapping)) or not isinstance(raw, (list, tuple)) or not raw:
        raise ConfigError(f"{where}: 'triage.budgets' must be a non-empty list of shares")
    values = []
    for entry in raw:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise ConfigError(f"{where}: budget {entry!r} is not a number")
        if not 0 < float(entry) <= 1:
            raise ConfigError(
                f"{where}: budget {entry!r} must be a share of the corpus in (0, 1], "
                "e.g. 0.01 for the first 1% a reviewer opens"
            )
        values.append(float(entry))
    return tuple(sorted(set(values)))


def _produced_by(raw: Any, where: Path) -> ProducedBy | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where}: 'produced_by' must be a mapping")
    model = raw.get("model")
    prompt_version = raw.get("prompt_version")
    if not isinstance(model, str) or not model:
        raise ConfigError(f"{where}: 'produced_by.model' is required — it is stamped on results")
    if not isinstance(prompt_version, str) or not prompt_version:
        raise ConfigError(
            f"{where}: 'produced_by.prompt_version' is required. Without it, two runs across "
            "a prompt change look comparable and are not."
        )
    extra = {k: v for k, v in raw.items() if k not in {"model", "prompt_version"}}
    return ProducedBy(model=model, prompt_version=prompt_version, extra=extra or None)


def _budget(raw: Any, where: Path) -> Budget:
    if raw is None:
        return Budget()
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where}: 'budget' must be a mapping")
    unknown = set(raw) - {"max_usd", "confirm"}
    if unknown:
        raise ConfigError(f"{where}: 'budget' has unknown key(s) {sorted(unknown)}")
    max_usd = raw.get("max_usd")
    if max_usd is not None:
        if isinstance(max_usd, bool) or not isinstance(max_usd, (int, float)) or max_usd < 0:
            raise ConfigError(f"{where}: 'budget.max_usd' must be a non-negative number")
        max_usd = float(max_usd)
    confirm = raw.get("confirm", True)
    if not isinstance(confirm, bool):
        raise ConfigError(f"{where}: 'budget.confirm' must be true or false")
    return Budget(max_usd=max_usd, confirm=confirm)
