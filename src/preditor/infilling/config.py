"""This module provides configuration for the infilling algorithms."""

import pydantic


class InfillingConfig(pydantic.BaseModel):
    """Configuration for the infilling algorithms.

    max_length: The maximum number of tokens generated during infilling.
    num_variants: The number of variants that infilling chooses from.
    """

    max_length: int = pydantic.Field(8, ge=1)
    num_variants: int = pydantic.Field(4, ge=2)
