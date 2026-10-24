"""Exceptions for expected, user-facing failures (bad ticker, no trained model, too little data).

The CLI prints these as a one-line ``error: ...`` instead of a traceback. Each one also
subclasses the matching built-in, so existing ``except ValueError`` handlers keep working.
Anything else raised by libraries or internal bugs still surfaces with a full traceback.
"""


class UserError(Exception):
    """Base class for failures caused by input or missing artifacts, not by a bug."""


class UserValueError(UserError, ValueError):
    pass


class ModelNotFoundError(UserError, FileNotFoundError):
    pass


class UserRuntimeError(UserError, RuntimeError):
    pass
