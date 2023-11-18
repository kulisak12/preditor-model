from typing import List, Set, Tuple

from prediktor import tags


class ReplacementVariantsGenerator:
    def __init__(
        self, text: str,
        start: int, length: int, replacement: str
    ) -> None:
        self.tagged_tokens = tags.tag(text)
        self._force_replacement(
            self.tagged_tokens, start, length, replacement
        )

    @staticmethod
    def _force_replacement(
        tagged_tokens: List[tags.TaggedToken],
        start: int, length: int, replacement: str
    ) -> None:
        """Force the form of the replaced token."""
        pos = 0
        for i, token in enumerate(tagged_tokens):
            if pos == start and len(token.form) == length:
                tagged_tokens[i] = tags.TaggedToken(
                    lemma=None,
                    tag=None,
                    form=replacement
                )
                return
            elif pos < start:
                pos += len(token.form)
            else:
                break
        raise ValueError(
            "The replacement is not a single token in the text."
        )

    def get_variants(
        self, prefix: str, num_prefix_tokens: int
    ) -> Tuple[Set[str], int]:
        """Extend given prefix with variants of following tokens."""
        variants = {prefix}
        while (
            num_prefix_tokens < len(self.tagged_tokens)
            and len(variants) == 1
        ):
            token = self.tagged_tokens[num_prefix_tokens]
            continuations = (
                {token.form} if token.lemma is None or token.tag is None
                else tags.generate_word_variations(token.lemma, token.tag)
            )

            num_prefix_tokens += 1
            current_prefix = next(iter(variants))
            variants = {
                current_prefix + continuation
                for continuation in continuations
            }
        return (variants, num_prefix_tokens)
