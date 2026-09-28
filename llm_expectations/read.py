"""Getting data in: jsonl, csv, parquet, and Python objects.

JSONL is the default because it streams — you can point at 500,000 items
without loading them, and every reader here is a generator so that stays true.
The typed adapters validate one row at a time and name the file and line when a
row is wrong, because "KeyError: 'item_id'" on row 40,112 of a file you did not
write is not a diagnosis.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

from .types import Item, Label, Output

__all__ = [
    "ReadError",
    "index_items",
    "read_items",
    "read_labels",
    "read_outputs",
    "read_rows",
]

Source = "str | Path | Iterable[Mapping[str, Any]] | Any"


class ReadError(ValueError):
    """A row could not be read. Always says where."""


def read_rows(source: Any, *, name: str = "input") -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Yield ``(where, row)`` for any supported source.

    ``where`` is what a person needs to go and look at the row: a file and a
    line number, or a position in the sequence.
    """
    if isinstance(source, (str, Path)):
        yield from _read_file(Path(source))
        return

    rows = _rows_from_object(source, name)
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ReadError(f"{name} row {position}: expected a mapping, got {type(row).__name__}")
        yield f"{name} row {position}", row


def _read_file(path: Path) -> Iterator[tuple[str, Mapping[str, Any]]]:
    if not path.exists():
        raise ReadError(f"no file at {path}")
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        yield from _read_jsonl(path)
    elif suffix == ".csv":
        yield from _read_csv(path)
    elif suffix == ".parquet":
        yield from _read_parquet(path)
    else:
        raise ReadError(
            f"{path}: unsupported extension {suffix!r}. Readable: .jsonl, .ndjson, .csv, "
            ".parquet — or pass a list of dicts or a dataframe directly."
        )


def _read_jsonl(path: Path) -> Iterator[tuple[str, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{path}:{number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReadError(f"{where}: not valid JSON — {exc.msg}") from None
            if not isinstance(row, Mapping):
                raise ReadError(f"{where}: expected an object, got {type(row).__name__}")
            yield where, row


def _read_csv(path: Path) -> Iterator[tuple[str, Mapping[str, Any]]]:
    # CSV has no types, so cells are coerced conservatively: a blank becomes
    # None, something that parses as a number becomes one, everything else
    # stays a string. This is lossy by nature, which is why jsonl is default.
    with path.open(encoding="utf-8", newline="") as handle:
        for number, row in enumerate(csv.DictReader(handle), start=2):
            yield f"{path}:{number}", {k: _coerce(v) for k, v in row.items() if k is not None}


def _coerce(value: str | None) -> Any:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _read_parquet(path: Path) -> Iterator[tuple[str, Mapping[str, Any]]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError:
        raise ReadError(
            f"{path} is parquet, which needs pyarrow: pip install 'llm-expectations[frames]'"
        ) from None
    table = parquet.read_table(path)
    for number, row in enumerate(table.to_pylist(), start=1):
        yield f"{path} row {number}", row


def _rows_from_object(source: Any, name: str) -> Iterable[Any]:
    # polars, then pandas, then anything iterable. Duck-typed on purpose: the
    # package does not import either one to read a frame someone already has.
    if hasattr(source, "to_dicts"):
        return cast("Iterable[Any]", source.to_dicts())
    if hasattr(source, "to_dict") and hasattr(source, "columns"):
        return cast("Iterable[Any]", source.to_dict(orient="records"))
    if isinstance(source, Iterable):
        return source
    raise ReadError(
        f"{name}: cannot read a {type(source).__name__}. Pass a path, a list of dicts, "
        "or a pandas/polars frame."
    )


def read_items(source: Any) -> Iterator[Item]:
    """Items: ``id`` and ``text`` are required, everything else is metadata."""
    for where, row in read_rows(source, name="items"):
        item_id = row.get("id", row.get("item_id"))
        if not isinstance(item_id, str) or not item_id:
            raise ReadError(f"{where}: an item needs a non-empty string 'id'")
        text = row.get("text")
        if text is None:
            raise ReadError(
                f"{where}: item {item_id!r} has no 'text'. An item with nothing to read is "
                "an exclusion, not a row — filter it out and the run will count it."
            )
        metadata = {k: v for k, v in row.items() if k not in {"id", "item_id", "text"}}
        yield Item(id=item_id, text=str(text), metadata=metadata)


def read_outputs(source: Any) -> Iterator[Output]:
    """Outputs: ``item_id`` plus the model's fields, side by side.

    Every other column is kept. The schema decides which are under test; the
    rest stay available to checks rather than being dropped at read time.
    """
    for where, row in read_rows(source, name="outputs"):
        item_id = row.get("item_id", row.get("id"))
        if not isinstance(item_id, str) or not item_id:
            raise ReadError(f"{where}: an output needs a non-empty string 'item_id'")
        values = {k: v for k, v in row.items() if k not in {"item_id", "id"}}
        yield Output(item_id=item_id, values=values)


def read_labels(source: Any) -> Iterator[Label]:
    """Human answers. Long-form — one row per ``(item, field, annotator)``."""
    for where, row in read_rows(source, name="labels"):
        missing = [k for k in ("item_id", "field", "label") if not row.get(k)]
        if missing:
            raise ReadError(f"{where}: a label row is missing {missing}")
        annotator = row.get("annotator")
        if not isinstance(annotator, str) or not annotator:
            raise ReadError(
                f"{where}: a label row needs an 'annotator'. It is what separates two "
                "people disagreeing from one person labelling twice, and that distinction "
                "is the whole of mode 2."
            )
        yield Label(
            item_id=str(row["item_id"]),
            field=str(row["field"]),
            label=str(row["label"]),
            annotator=annotator,
        )


def index_items(source: Any) -> dict[str, Item]:
    """Materialise items by id, refusing duplicates.

    The one place streaming is given up, because joining outputs to items needs
    random access. Two rows sharing an id means one of them is silently
    invisible from here on.
    """
    items: dict[str, Item] = {}
    for item in read_items(source):
        if item.id in items:
            raise ReadError(
                f"duplicate item id {item.id!r}. One of the two would be silently unused "
                "in every join from here on."
            )
        items[item.id] = item
    return items
