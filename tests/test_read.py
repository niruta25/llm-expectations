"""Readers: the formats, and saying where a bad row is."""

from __future__ import annotations

import pytest

from llm_expectations.read import (
    ReadError,
    index_items,
    read_items,
    read_labels,
    read_outputs,
    read_rows,
)

from .conftest import EXAMPLE


def write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestJsonl:
    def test_blank_lines_are_skipped(self, tmp_path):
        path = write(tmp_path, "i.jsonl", '{"id":"a","text":"x"}\n\n{"id":"b","text":"y"}\n')
        assert [i.id for i in read_items(path)] == ["a", "b"]

    def test_a_broken_line_is_named_by_file_and_line(self, tmp_path):
        path = write(tmp_path, "i.jsonl", '{"id":"a","text":"x"}\n{oops\n')
        with pytest.raises(ReadError, match=r"i\.jsonl:2: not valid JSON"):
            list(read_items(path))

    def test_reading_streams_rather_than_materialising(self, tmp_path):
        # The first row must arrive without the last one being parsed, which is
        # what lets this be pointed at 500,000 items.
        path = write(tmp_path, "i.jsonl", '{"id":"a","text":"x"}\n{broken\n')
        rows = read_items(path)
        assert next(rows).id == "a"
        with pytest.raises(ReadError):
            next(rows)


class TestCsv:
    def test_numbers_are_coerced_and_blanks_become_none(self, tmp_path):
        path = write(tmp_path, "o.csv", "item_id,jtbd,confidence,note\ns-1,billing,0.82,\n")
        output = next(read_outputs(path))
        assert output.values == {"jtbd": "billing", "confidence": 0.82, "note": None}

    def test_the_row_number_matches_what_a_spreadsheet_shows(self, tmp_path):
        path = write(tmp_path, "i.csv", "id,text\n,nothing\n")
        with pytest.raises(ReadError, match=r"i\.csv:2"):
            list(read_items(path))


class TestPythonObjects:
    def test_a_list_of_dicts_needs_no_file(self):
        assert [i.id for i in read_items([{"id": "a", "text": "x"}])] == ["a"]

    def test_a_polars_style_frame_is_duck_typed(self):
        class Frame:
            def to_dicts(self):
                return [{"id": "a", "text": "x"}]

        assert [i.id for i in read_items(Frame())] == ["a"]

    def test_a_pandas_style_frame_is_duck_typed(self):
        class Frame:
            columns = ["id", "text"]

            def to_dict(self, orient):
                assert orient == "records"
                return [{"id": "a", "text": "x"}]

        assert [i.id for i in read_items(Frame())] == ["a"]

    def test_a_position_is_reported_when_there_is_no_line_number(self):
        with pytest.raises(ReadError, match="items row 2"):
            list(read_items([{"id": "a", "text": "x"}, {"id": "b"}]))


class TestRequiredColumns:
    def test_an_item_with_no_text_is_an_exclusion_not_a_row(self):
        with pytest.raises(ReadError, match="exclusion, not a row"):
            list(read_items([{"id": "a"}]))

    def test_item_id_is_accepted_as_an_alias_for_id(self):
        assert next(read_items([{"item_id": "a", "text": "x"}])).id == "a"

    def test_everything_not_named_is_kept(self):
        item = next(read_items([{"id": "a", "text": "x", "channel": "chat"}]))
        assert item.metadata == {"channel": "chat"}
        output = next(read_outputs([{"item_id": "a", "jtbd": "b", "prompt_id": "p8"}]))
        assert output.values == {"jtbd": "b", "prompt_id": "p8"}

    def test_a_label_without_an_annotator_is_refused(self):
        # It is what separates two people disagreeing from one person labelling
        # twice, and that distinction is the whole of mode 2.
        with pytest.raises(ReadError, match="whole of mode 2"):
            list(read_labels([{"item_id": "a", "field": "jtbd", "label": "b"}]))


def test_a_duplicate_item_id_is_refused_rather_than_silently_shadowed():
    with pytest.raises(ReadError, match="duplicate item id"):
        index_items([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}])


def test_an_unreadable_extension_lists_what_is_readable(tmp_path):
    path = write(tmp_path, "i.txt", "whatever")
    with pytest.raises(ReadError, match=r"\.jsonl, \.ndjson, \.csv, \.parquet"):
        list(read_rows(path))


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(ReadError, match="no file at"):
        list(read_rows(tmp_path / "gone.jsonl"))


def test_the_worked_example_reads():
    items = index_items(EXAMPLE / "items.jsonl")
    outputs = {o.item_id: o for o in read_outputs(EXAMPLE / "outputs.jsonl")}
    labels = list(read_labels(EXAMPLE / "labels.jsonl"))
    assert len(items) == len(outputs) == 13
    assert set(outputs) == set(items)
    assert {label.annotator for label in labels} == {"ann-1", "ann-2"}
