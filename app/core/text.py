"""Text sanitisation helpers.

Bytes that cannot be decoded as UTF-8 (e.g. filenames on a Windows mount, or
terminal input under a non-UTF-8 codepage) are turned by Python into lone
surrogates in the range U+DC80-U+DCFF. Those surrogates are not valid Unicode
and raise ``UnicodeEncodeError`` when printed or written as UTF-8.

``clean_text`` round-trips such strings through ``surrogateescape`` so the stray
byte is recovered, then decodes with ``replace`` to drop it cleanly.
"""


def clean_text(value):
    """Return ``value`` free of lone surrogates (no-op for valid text)."""
    if not isinstance(value, str):
        return value
    try:
        return value.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    except UnicodeEncodeError:
        # Lone high surrogates are not handled by surrogateescape; fall back.
        return value.encode("utf-8", "replace").decode("utf-8", "replace")


def clean_data(value):
    """Recursively sanitise strings inside dicts, lists and tuples."""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, dict):
        return {
            (clean_text(k) if isinstance(k, str) else k): clean_data(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [clean_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clean_data(item) for item in value)
    return value