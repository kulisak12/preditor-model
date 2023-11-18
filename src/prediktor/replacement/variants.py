from typing import List, Set, Tuple

from prediktor import tags


class ReplacementVariantsGenerator:
    def __init__(
        self, text: str,
        start: int, length: int, replacement: str
    ) -> None:
        self.tagged_forms = tags.tag(text)
        self._force_replacement(
            self.tagged_forms, start, length, replacement
        )

    @staticmethod
    def _force_replacement(
        tagged_forms: List[tags.TaggedForm],
        start: int, length: int, replacement: str
    ) -> None:
        """Force the form of the replaced form."""
        pos = 0
        for i, form in enumerate(tagged_forms):
            if pos == start and len(form.form) == length:
                tagged_forms[i] = tags.TaggedForm(
                    lemma=None,
                    tag=None,
                    form=replacement
                )
                return
            elif pos < start:
                pos += len(form.form)
            else:
                break
        raise ValueError(
            "The replacement is not a single word in the text."
        )

    def get_variants(
        self, num_prefix_forms: int
    ) -> Tuple[Set[str], int]:
        """Construct possible extensions of a prefix of the text.

        Returns a set of possible extensions and the number of forms in
        the extensions (same for all of them).
        """
        variants = {""}
        next_form_index = num_prefix_forms
        while (
            next_form_index < len(self.tagged_forms)
            and len(variants) == 1
        ):
            form = self.tagged_forms[next_form_index]
            continuations = tags.generate_word_variations(form)

            next_form_index += 1
            current_prefix = next(iter(variants))
            variants = {
                current_prefix + continuation
                for continuation in continuations
            }
        return (variants, next_form_index - num_prefix_forms)
