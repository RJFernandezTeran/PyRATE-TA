"""Logging for PyRATE-TA.

Every module obtains its logger with :func:`get_logger`, so all records travel
under the single ``pyrate_ta`` logger and can be silenced, redirected or raised in
verbosity by the host application::

    import logging
    logging.getLogger("pyrate_ta").setLevel(logging.DEBUG)   # full detail
    logging.getLogger("pyrate_ta").setLevel(logging.WARNING) # quiet

PyRATE-TA is used both as a library and as an application (GUI, console scripts),
so :func:`configure_logging` attaches a plain console handler on import *only*
when nothing else has configured logging. A host that sets up its own handlers
-- on the root logger or on ``pyrate_ta`` -- keeps full control and nothing is
added. This matters more here than in a standalone package: PyMORGAN configures
a ``pymorgan`` logger the same way, and the two must not fight over the root
logger when both are imported in one process.

Three conventions are used throughout the code base:

* ``logger.info`` for messages the user is meant to see -- fit summaries,
  recovered lifetimes, convergence status, method citations;
* ``logger.warning`` for results that are suspect but returned anyway
  (near-degenerate lifetimes, parameters pinned at a bound);
* ``logger.debug(..., exc_info=True)`` inside ``except`` blocks whose failure is
  genuinely recoverable, so the traceback is retrievable at DEBUG level instead
  of being discarded.

A fit that did *not* converge is never merely logged: it raises, or the result
object carries the failure explicitly (see ``pyrate_ta.results``).
"""

from __future__ import annotations

import logging

LOGGER_NAME = "pyrate_ta"

# Bare message: these records are read by scientists at a console, not parsed.
_FORMAT = "%(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the ``pyrate_ta`` logger, or a child of it.

    ``get_logger(__name__)`` from inside the package yields the dotted module
    logger (``pyrate_ta.fit.global_analysis``); any other name is attached as a
    child.
    """
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(level: int | str = logging.INFO, *, force: bool = False) -> logging.Logger:
    """Give the ``pyrate_ta`` logger a console handler, unless one is set up.

    Called once on import so that informational output keeps reaching the
    console. If the root logger or the ``pyrate_ta`` logger already has handlers
    (i.e. the host application configured logging) nothing is changed, unless
    ``force=True``.
    """
    logger = logging.getLogger(LOGGER_NAME)
    configured = bool(logger.handlers) or bool(logging.getLogger().handlers)
    if configured and not force:
        return logger
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


#: Width of the rule drawn between fit runs.
_RULE_WIDTH = 72


def log_run_header(title: str, *, logger=None) -> str:
    """Draw a rule and a title, so consecutive fits are told apart.

    Several fits in one session otherwise run together in the console, and the
    lifetimes of the previous one read as part of the next. Returns the text so
    a caller can reuse it.
    """
    log = logger if logger is not None else get_logger(__name__)
    rule = "=" * _RULE_WIDTH
    text = f"\n{rule}\n  {title}\n{rule}"
    log.info("%s", text)
    return text
