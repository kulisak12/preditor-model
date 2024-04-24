import pytest

from preditor.suggestion import suggestion


@pytest.mark.parametrize("input_lines, expected_output", [
    ([], []),
    ([""], []),
    (["", "First line."], ["First line."]),
    (["First line.", "Second line.", ""], ["First line.", "Second line."]),
    (["First line.", "", "Second line."], ["First line."]),
    (["First line.", "Second line.", "Third line."], ["First line.", "Second line.", "Third line."]),
    (["", "First line.", "Second line."], ["First line.", "Second line."])
])
def test_get_first_paragraph(input_lines, expected_output):
    assert suggestion._get_first_paragraph(input_lines) == expected_output


@pytest.mark.parametrize("input_lines, expected_output", [
    ([], []),
    ([""], []),
    (["", "First line."], ["First line."]),
    (["First line.", "Second line.", ""], ["First line.", "Second line."]),
    (["First line.", "", "Second line."], ["Second line."]),
    (["First line.", "Second line.", "Third line."], ["First line.", "Second line.", "Third line."]),
    (["", "First line.", "Second line."], ["First line.", "Second line."])
])
def test_get_last_paragraph(input_lines, expected_output):
    assert suggestion._get_last_paragraph(input_lines) == expected_output
