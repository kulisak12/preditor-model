import pytest

from preditor.substitution import substitution


@pytest.mark.parametrize(
    "before_old, old, after_old, expected_previous, expected_sentence, expected_next",
    [
        ("Ahoj Pepo! Mám pro ", "tebe", " překvapení. Hádej, co to je.",
         "Ahoj Pepo! ", "Mám pro tebe překvapení.", " Hádej, co to je."),
        ("Ahoj Pepo! ", "Mám", " pro tebe překvapení. Hádej, co to je.",
         "Ahoj Pepo! ", "Mám pro tebe překvapení.", " Hádej, co to je."),
        ("Ahoj ", "Pepo", "! Mám pro tebe překvapení. Hádej, co to je.",
         "", "Ahoj Pepo!", " Mám pro tebe překvapení. Hádej, co to je."),
        ("Ahoj Pepo! Mám pro tebe překvapení. Hádej, ", "co", " to je.",
         "Ahoj Pepo! Mám pro tebe překvapení. ", "Hádej, co to je.", ""),
    ],
)
def test_find_sentence_with_old(
    before_old, old, after_old,
    expected_previous, expected_sentence, expected_next
):
    previous_sentences, sentence, next_sentences = substitution._find_sentence_with_old(
        before_old, old, after_old
    )
    assert previous_sentences == expected_previous
    assert sentence == expected_sentence
    assert next_sentences == expected_next
