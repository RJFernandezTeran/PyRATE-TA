"""Small shared helpers. No Qt, no PyMORGAN, no scipy.

Currently the parsing of lifetimes typed by a user, which both the GUI table
and any console entry point go through, so that "inf" means the same thing
everywhere.
"""

from __future__ import annotations

import math

from .log import get_logger

logger = get_logger(__name__)

# Spellings accepted for an infinite (non-decaying) lifetime. Compared after
# lower-casing and stripping, so "INF", " Infinite " and "1e999" all arrive
# here as the same thing.
_INFINITE_TOKENS = frozenset(
    {
        "inf",
        "+inf",
        "infty",
        "+infty",
        "infinity",
        "+infinity",
        "infinite",
        "∞",  # the infinity sign, as typed on a Mac or pasted from a paper
        "+∞",
        "oo",
        "const",
        "constant",
        "offset",
        "fixed",
        "nondecaying",
        "non-decaying",
    }
)


def is_infinite_lifetime(value) -> bool:
    """Whether ``value`` means "this component does not decay".

    Accepts a float (``inf``) or any of the spellings a user might type
    (``inf``, ``Infinity``, ``∞``, ``constant``, ``offset``, ...).
    """
    if isinstance(value, str):
        return value.strip().lower().replace(" ", "") in _INFINITE_TOKENS
    try:
        return math.isinf(float(value))
    except (TypeError, ValueError):
        return False


def parse_lifetime(value, *, field: str = "lifetime") -> float:
    """Parse a lifetime typed by a user into a float.

    An infinite lifetime is a legitimate entry, not an error: it adds a
    non-decaying component -- the constant offset that accounts for signal
    outliving the measured delay window. It is returned as ``math.inf``, whose
    rate is zero, and it is always treated as a **fixed** parameter (an
    optimiser cannot vary infinity).

    Parameters
    ----------
    value : str or float
        The entry. Comma decimal separators are accepted, so "1,5" reads as
        1.5 -- typing a European decimal comma should not silently produce 15.
    field : str
        Name used in the error message.

    Returns
    -------
    float
        A finite positive lifetime, or ``math.inf``.

    Raises
    ------
    ValueError
        If the entry is not a number, or is zero or negative. A lifetime of
        zero has no meaning and is refused rather than clamped.
    """
    if is_infinite_lifetime(value):
        return math.inf
    if isinstance(value, str):
        text = value.strip().replace(" ", "")
        if not text:
            raise ValueError(f"{field} is empty")
        # A comma is a decimal separator here; thousands separators are not
        # used in this field, so there is no ambiguity to resolve.
        text = text.replace(",", ".")
        try:
            number = float(text)
        except ValueError:
            raise ValueError(f"could not read {field} {value!r} as a number") from None
    else:
        number = float(value)

    if math.isnan(number):
        raise ValueError(f"{field} is not a number")
    if number <= 0:
        raise ValueError(f"{field} must be positive, got {number!r}")
    return number


def format_lifetime(value, fmt: str = "%.4g") -> str:
    """Render a lifetime for display, showing an infinite one as ``inf``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(number):
        return "inf"
    return fmt % number
