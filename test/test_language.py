import pytest

from preditor import language

LANGS = ["en", "cs"]


@pytest.mark.parametrize("text, expected", [
    ("Zákon, který tohle nerozpozná, je podle mě špatný.", "cs"),
    ("It was a truly free country for only twenty years.", "en"),
    ("Mesto je plné života a žiadne Vianoce tu nie sú.", "cs"),
])
def test_estimate_language(text, expected):
    assert language.estimate_language(text, LANGS) == expected
