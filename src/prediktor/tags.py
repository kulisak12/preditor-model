import dataclasses
from typing import List, Set

from ufal import morphodita

from prediktor.config import Config

tagger = morphodita.Tagger.load(Config.dict_path)
if not tagger:
    raise Exception(f"Cannot load tagger from file '{Config.dict_path}'.")


@dataclasses.dataclass
class TaggedToken:
    lemma: str
    tag: str
    start: int
    length: int

GUESSER = morphodita.Morpho.GUESSER


def tag(text: str) -> List[TaggedToken]:
    """Tag the given text using the loaded tagger."""
    forms = morphodita.Forms()
    lemmas = morphodita.TaggedLemmas()
    tokens = morphodita.TokenRanges()
    tokenizer = tagger.newTokenizer()
    if tokenizer is None:
        raise Exception("No tokenizer is defined for the supplied model!")

    result: List[TaggedToken] = []
    tokenizer.setText(text)
    while tokenizer.nextSentence(forms, tokens):
        tagger.tag(forms, lemmas, GUESSER)
        for lemma, token in zip(lemmas, tokens):
            tagged = TaggedToken(
                lemma=lemma.lemma,
                tag=lemma.tag,
                start=token.start,
                length=token.length,
            )
            result.append(tagged)
    return result


def generate_word_variations(lemma: str, tag: str) -> Set[str]:
    """Generate lemma forms to consider when changing a sentence.

    Only meaningful variations are considered.
    The original lemma will always be included in the result.
    """
    morpho = tagger.getMorpho()
    wildcard = create_tag_wildcard(tag)
    lemmas_forms = morphodita.TaggedLemmasForms()
    morpho.generate(lemma, wildcard, GUESSER, lemmas_forms)
    assert len(lemmas_forms) == 1
    return {
        form.form
        for lemma_forms in lemmas_forms
        for form in lemma_forms.forms
    }


def create_tag_wildcard(tag: str) -> str:
    """Create a tag wildcard to generate lemma forms to consider."""
    # 1  pos
    # 2  subpos
    # 3  gender
    # 4  number
    # 5  case
    # 6  possgender
    # 7  possnumber
    # 8  person
    # 9  tense
    # 10 grade
    # 11 negation
    # 12 voice
    # 13 aspect
    # 14 aggregate
    # 15 var
    wildcard_positions = [3, 4, 6, 7, 8]
    tag_chars = list(" " + tag)  # shift to one-based indexing
    for i in wildcard_positions:
        tag_chars[i] = "?"
    if tag_chars[14] != "-":
        tag_chars[14] = "?"
    variants = "".join({ "-", "1", tag_chars[15] })
    tag_chars[15] = "[" + variants + "]"
    wildcard = "".join(tag_chars)
    return wildcard[1:]  # shift back
