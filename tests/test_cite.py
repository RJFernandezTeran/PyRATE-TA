"""Tests for :mod:`pyrate_ta.cite`.

The citation text is checked for the DOIs themselves, not for wording: a typo in
a DOI is the failure that matters, since it sends the reader to the wrong paper.
"""

from __future__ import annotations

import pytest

import pyrate_ta as pr
from pyrate_ta.cite import REFERENCES, citation_text, cite, get_reference, reset_citations


@pytest.fixture(autouse=True)
def _fresh_session():
    reset_citations()
    yield
    reset_citations()


def test_every_reference_is_complete():
    for key, ref in REFERENCES.items():
        assert ref.key == key
        assert ref.citation.strip()
        assert ref.what.strip()


@pytest.mark.parametrize(
    ("key", "doi"),
    [
        ("chameleon2022", "10.1021/acs.jchemed.2c00104"),
        ("vanstokkum2004", "10.1016/j.bbabio.2004.04.011"),
        ("berberansantos1990", "10.1021/ed067p375"),
        ("kovalenko1999", "10.1103/PhysRevA.59.2369"),
        ("optimus2015", "10.1021/ac504348h"),
        ("hansen1992", "10.1137/1034115"),
    ],
)
def test_dois_are_exact(key, doi):
    assert get_reference(key).doi == doi


def test_unpublished_paper_has_no_invented_doi():
    """The forthcoming paper must show a gap, never a made-up identifier."""
    ref = get_reference("pyrate_ta")
    assert ref.doi is None
    assert "in preparation" in str(ref)


def test_citation_text_lists_everything():
    text = citation_text()
    for ref in REFERENCES.values():
        assert ref.citation.split(",")[0] in text
    assert "please cite" in text
    assert pr.__version__ in text


def test_cite_logs_once_per_session(caplog):
    with caplog.at_level("DEBUG", logger="pyrate_ta.cite"):
        first = cite("vanstokkum2004")
        second = cite("vanstokkum2004")
    assert len(first) == 1 and second == []
    assert caplog.text.count("10.1016/j.bbabio.2004.04.011") == 1


def test_a_fit_does_not_reprint_the_references(caplog):
    """The banner at start-up is enough; a run should show its own result."""
    import numpy as np

    from pyrate_ta.cite import reset_citations

    reset_citations()
    t = np.linspace(0.0, 50.0, 60)
    D = np.exp(-t[:, None] / 10.0) * np.linspace(1.0, 2.0, 8)[None, :]
    with caplog.at_level("INFO", logger="pyrate_ta"):
        pr.fit_global(D, t, taus=[8.0], model_type="Sequential")
    assert "10.1016/j.bbabio.2004.04.011" not in caplog.text
    assert "=" * 20 in caplog.text  # ... but the run is separated by a rule


def test_cite_can_repeat_when_asked(caplog):
    with caplog.at_level("DEBUG", logger="pyrate_ta.cite"):
        cite("berberansantos1990")
        again = cite("berberansantos1990", once=False)
    assert len(again) == 1


def test_unknown_key_is_rejected():
    with pytest.raises(KeyError):
        cite("no_such_paper")


def test_eigen_propagation_cites_its_method(caplog):
    """A routine implementing a published method names its source.

    At *debug* level: the start-up banner already says what to cite, and a
    reference repeated inside every fit buries the result it accompanies.
    """
    from pyrate_ta.models import SequentialModel

    with caplog.at_level("DEBUG", logger="pyrate_ta"):
        SequentialModel(n_components=2).concentrations([0.0, 1.0, 2.0], taus=[1.0, 10.0])
    assert "10.1021/ed067p375" in caplog.text  # eigenvector solution
    assert "10.1016/j.bbabio.2004.04.011" in caplog.text  # EAS/SAS formalism


def test_parallel_model_does_not_claim_target_analysis(caplog):
    """A DAS fit is not global/target analysis, so it must not cite it."""
    from pyrate_ta.models import ParallelModel

    with caplog.at_level("DEBUG", logger="pyrate_ta"):
        ParallelModel(n_components=2).concentrations([0.0, 1.0], taus=[1.0, 10.0])
    assert "10.1016/j.bbabio.2004.04.011" not in caplog.text
