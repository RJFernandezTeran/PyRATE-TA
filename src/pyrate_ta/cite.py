"""Literature references and the "please cite" notice.

Two mechanisms, both going through the ``pyrate`` logger rather than ``print``
(the console entry points are the exception; see :func:`print_citations`):

* :func:`citation_notice` -- the start-up banner listing what to cite when
  results obtained with PyRATE-TA are published.
* :func:`cite` -- the per-method citation a routine records when it finishes,
  so the formalism actually used is traceable. It goes to the *debug* log: the
  start-up banner already says what to cite, and a reference printed in the
  middle of every fit only buries the result. Each key is recorded once per
  session in any case.

Every entry was checked against Crossref. The forthcoming PyRATE-TA / PyMORGAN paper
is listed with its DOI missing on purpose -- an invented DOI is worse than a
visible gap -- and must be completed when it appears.
"""

from __future__ import annotations

from dataclasses import dataclass

from .log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Reference:
    """One literature reference.

    Attributes
    ----------
    key : str
        Short identifier used by :func:`cite`.
    citation : str
        Human-readable citation, author-year style.
    doi : str or None
        DOI, or ``None`` while a paper is still unpublished.
    what : str
        What in PyRATE-TA this reference covers, so a log line explains itself.
    """

    key: str
    citation: str
    doi: str | None
    what: str

    def __str__(self) -> str:
        tail = f" DOI: {self.doi}" if self.doi else " (manuscript in preparation)"
        return f"{self.citation}{tail}"


REFERENCES: dict[str, Reference] = {
    "pyrate_ta": Reference(
        key="pyrate_ta",
        citation=(
            "R. J. Fernandez-Teran, 'PyRATE-TA: Rate Analysis & Target-model Engine for Transient Absorption' "
            "and 'PyMORGAN'."
        ),
        doi=None,
        what="the software itself",
    ),
    "chameleon2022": Reference(
        key="chameleon2022",
        citation=(
            "R. J. Fernandez-Teran, E. Sucre-Rosales, L. Echevarria, F. E. Hernandez, "
            "'A Sweet Introduction to the Mathematical Analysis of Time-Resolved Spectra "
            "and Complex Kinetic Mechanisms: The Chameleon Reaction Revisited', "
            "J. Chem. Educ. 2022, 99, 2327-2337."
        ),
        doi="10.1021/acs.jchemed.2c00104",
        what="the analysis framework PyRATE-TA implements",
    ),
    "vanstokkum2004": Reference(
        key="vanstokkum2004",
        citation=(
            "I. H. M. van Stokkum, D. S. Larsen, R. van Grondelle, "
            "'Global and target analysis of time-resolved spectra', "
            "Biochim. Biophys. Acta Bioenerg. 2004, 1657, 82-104."
        ),
        doi="10.1016/j.bbabio.2004.04.011",
        what="global and target analysis, and the variable-projection formalism",
    ),
    "berberansantos1990": Reference(
        key="berberansantos1990",
        citation=(
            "M. N. Berberan-Santos, J. M. G. Martinho, "
            "'The integration of kinetic rate equations by matrix methods', "
            "J. Chem. Educ. 1990, 67, 375."
        ),
        doi="10.1021/ed067p375",
        what="the eigenvector solution of coupled first-order systems",
    ),
    "kovalenko1999": Reference(
        key="kovalenko1999",
        citation=(
            "S. A. Kovalenko, A. L. Dobryakov, J. Ruthmann, N. P. Ernsting, "
            "'Femtosecond spectroscopy of condensed phases with chirped supercontinuum "
            "probing', Phys. Rev. A 1999, 59, 2369-2384."
        ),
        doi="10.1103/PhysRevA.59.2369",
        what="the Gaussian-and-derivatives description of the coherent artefact",
    ),
    "optimus2015": Reference(
        key="optimus2015",
        citation=(
            "C. Slavov, H. Hartmann, J. Wachtveitl, "
            "'Implementation and Evaluation of Data Analysis Strategies for Time-Resolved Optical Spectroscopy', "
            "Anal. Chem. 2015, 87, 2328-2336."
        ),
        doi="10.1021/ac504348h",
        what="Lifetime Density Analysis (LDA) and the OPTIMUS analysis framework",
    ),
    "hansen1992": Reference(
        key="hansen1992",
        citation=(
            "P. C. Hansen, "
            "'Analysis of discrete ill-posed problems by means of the L-curve', "
            "SIAM Review 1992, 34, 561-580."
        ),
        doi="10.1137/1034115",
        what="L-curve corner detection for automatic regularisation parameter selection in LDA",
    ),
}

# Cited on start-up, in this order: the software paper first, then the paper
# describing the analysis framework it implements.
_STARTUP_KEYS = ("pyrate_ta", "chameleon2022")

# Keys already logged in this session, so a repeated fit does not repeat the
# reference. Cleared by :func:`reset_citations`, which the tests use.
_LOGGED: set[str] = set()


def get_reference(key: str) -> Reference:
    """Look up one reference, listing the alternatives on failure."""
    try:
        return REFERENCES[key]
    except KeyError:
        raise KeyError(
            f"unknown reference {key!r}; available: {', '.join(sorted(REFERENCES))}"
        ) from None


def cite(*keys: str, once: bool = True) -> list[Reference]:
    """Log the reference(s) for a method that has just run.

    Parameters
    ----------
    *keys
        Reference keys, e.g. ``"vanstokkum2004"``.
    once
        Skip keys already logged in this session (the default). Pass ``False``
        to log unconditionally.

    Returns
    -------
    list of Reference
        The references actually logged, so a caller can attach them to a fit
        result.
    """
    emitted: list[Reference] = []
    for key in keys:
        ref = get_reference(key)
        if once and key in _LOGGED:
            continue
        _LOGGED.add(key)
        # Debug, not info: the start-up banner already lists what to cite, and
        # repeating a reference in the middle of every fit buries the result it
        # is meant to accompany. The record is kept -- the keys are still
        # collected, and -v shows them -- it is simply not shouted per run.
        logger.debug("Method reference (%s): %s", ref.what, ref)
        emitted.append(ref)
    return emitted


def reset_citations() -> None:
    """Forget which references have been logged (used by the tests)."""
    _LOGGED.clear()


def citation_text(keys: tuple[str, ...] | list[str] | None = None) -> str:
    """The "please cite" block, as a plain multi-line string.

    ``keys`` defaults to every reference: the software paper, then the paper
    describing the framework, then the two method references.
    """
    from .__about__ import __version__

    order = (
        list(keys)
        if keys is not None
        else list(_STARTUP_KEYS) + [k for k in REFERENCES if k not in _STARTUP_KEYS]
    )
    lines = [
        f"PyRATE-TA v{__version__} - if you publish results obtained with this software,",
        "please cite:",
    ]
    lines += [f"  [{i + 1}] {get_reference(k)}" for i, k in enumerate(order)]
    return "\n".join(lines)


def citation_notice(keys: tuple[str, ...] | list[str] | None = None) -> str:
    """Log the citation block through the ``pyrate`` logger and return it.

    Called by the console entry points at start-up. It goes to ``logger.info``
    rather than ``print`` so that a host application which has configured
    logging keeps control of where it appears.
    """
    text = citation_text(keys)
    logger.info("%s", text)
    return text


def print_citations(keys: tuple[str, ...] | list[str] | None = None) -> None:
    """Print the citation block to the console.

    One of the few sanctioned uses of ``print``: this is an explicit
    ``print_*`` helper for interactive use (``python -m pyrate.cite``).
    """
    print(citation_text(keys))


if __name__ == "__main__":  # pragma: no cover - console helper
    print_citations()
