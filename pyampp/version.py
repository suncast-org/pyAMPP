"""Package version helper."""

try:
    from ._version import version
except Exception:
    # Fallback for editable/local source trees where _version.py is absent.
    version = "1.0.4"
