import dataclasses
from typing import Iterator, List, Optional, Set, Tuple

from ufal import morphodita

from preditor.config import Config

tagger = morphodita.Tagger.load(Config.dict_path)
if not tagger:
    raise Exception(f"Cannot load tagger from file '{Config.dict_path}'.")


@dataclasses.dataclass(frozen=True)
class TaggedForm:
    lemma: Optional[str]
    tag: Optional[str]
    form: str


GUESSER = morphodita.Morpho.GUESSER


def tag(text: str) -> List[TaggedForm]:
    """Tag the given text using the loaded tagger."""
    lemmas = morphodita.TaggedLemmas()  # type: ignore[abstract]
    result: List[TaggedForm] = []
    text_pos = 0
    for forms, tokens in tokenize(text):
        tagger.tag(forms, lemmas, GUESSER)
        for lemma, token in zip(lemmas, tokens):
            if token.start != text_pos:
                result.append(TaggedForm(
                    lemma=None,
                    tag=None,
                    form=text[text_pos:token.start]
                ))
            result.append(TaggedForm(
                lemma=lemma.lemma,
                tag=lemma.tag,
                form=text[token.start:token.start+token.length]
            ))
            text_pos = token.start + token.length
    return result


def split_sentences(text: str) -> List[str]:
    """Split the text into sentences.

    Return a list of sentences and text parts between them.
    """
    result: List[str] = []
    text_pos = 0
    for forms, tokens in tokenize(text):
        if len(tokens) == 0:
            continue
        sentence_start = tokens[0].start
        sentence_end = tokens[-1].start + tokens[-1].length
        between_sentences = text[text_pos:sentence_start]
        sentence = text[sentence_start:sentence_end]
        text_pos = sentence_end
        if between_sentences:
            result.append(between_sentences)
        result.append(sentence)
    return result


def tokenize(text: str) -> Iterator[Tuple[morphodita.Forms, morphodita.TokenRanges]]:
    """Tokenize the text.

    Generate sentences, each represented by a pair of forms and tokens.
    """
    forms = morphodita.Forms()  # type: ignore[abstract]
    tokens = morphodita.TokenRanges()  # type: ignore[abstract]
    tokenizer = tagger.newTokenizer()
    if tokenizer is None:
        raise Exception("No tokenizer is defined for the supplied model!")

    tokenizer.setText(text)
    while tokenizer.nextSentence(forms, tokens):
        yield (forms, tokens)


def generate_word_variations(original: TaggedForm) -> Set[str]:
    """Generate lemma forms to consider when changing a sentence.

    Only meaningful variations are considered.
    The original form will always be included in the result.
    """
    original_result = {original.form}
    if original.lemma is None or original.tag is None:
        return original_result
    morpho = tagger.getMorpho()
    wildcard = create_tag_wildcard(original.tag)
    lemmas_forms = morphodita.TaggedLemmasForms()  # type: ignore[abstract]
    morpho.generate(original.lemma, wildcard, GUESSER, lemmas_forms)
    variations = {
        copy_case(form.form, original.form)
        for lemma_forms in lemmas_forms
        for form in lemma_forms.forms
    }
    return variations | original_result


def copy_case(word: str, reference: str) -> str:
    """Copy the case from the reference word to the word."""
    if reference.isupper():
        return word.upper()
    if reference[0].isupper():
        return word[0].upper() + word[1:]
    return word


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
    tag_chars[15] = "[-1]"
    wildcard = "".join(tag_chars)
    return wildcard[1:]  # shift back
