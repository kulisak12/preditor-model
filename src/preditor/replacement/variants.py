from typing import List, Set, Tuple

from preditor import tags


class ReplacementVariantsGenerator:
    def __init__(
        self, text: str,
        start: int, length: int, replacement: str
    ) -> None:
        self._tagged_forms = tags.tag(text)
        self._force_replacement(start, length, replacement)
        self._variants = [
            tags.generate_word_variations(form)
            for form in self._tagged_forms
        ]

    def _force_replacement(
        self, start: int, length: int, replacement: str
    ) -> None:
        """Force the form of the replaced form."""
        pos = 0
        for i, form in enumerate(self._tagged_forms):
            if pos == start and len(form.form) == length:
                self._tagged_forms[i] = tags.TaggedForm(
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

    @property
    def num_forms(self) -> int:
        return len(self._tagged_forms)

    def get_extensions(
        self, extension_begin: int, min_variants: int = 2
    ) -> Tuple[Set[str], int]:
        """Construct possible extensions of a prefix of the text.

        extension_begin:
            The index of the form where the extension begins.
        min_variants:
            The minimum number of variants to construct.
            There will be fewer if the end of the text is reached.

        Return a tuple containing a set of possible extensions
        and the index of the end of the extension (exclusive).
        """
        min_variants = max(min_variants, 2)  # at least 2 variants needed
        extension_end = self._find_extension_end(extension_begin, min_variants)
        extensions = {""}
        for variants in self._variants[extension_begin:extension_end]:
            extensions = {
                extension + variant
                for variant in variants
                for extension in extensions
            }
        return (extensions, extension_end)

    def _find_extension_end(
        self, extension_begin: int, min_variants: int
    ) -> int:
        """Find the index of the end of the extension (exclusive).

        Once the required number of variants is reached,
        extend the variants as long as the number of them does not increase.
        Ensure that the extensions do not end with whitespace.
        """
        last_non_whitespace = extension_begin
        total_variants = 1
        for i in range(extension_begin, len(self._variants)):
            variants_here = self._variants[i]
            if total_variants >= min_variants and len(variants_here) > 1:
                return last_non_whitespace + 1
            total_variants *= len(variants_here)
            if self._tagged_forms[i].form.strip():
                last_non_whitespace = i
        return len(self._tagged_forms)
