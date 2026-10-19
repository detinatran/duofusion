"""HemoPIC+: extensions of HemoPIC (MICCAI 2026) for cerebral perfusion."""
import sys


def utf8_console():
    """Windows consoles/pipes default to cp1252, which crashes on non-ASCII paths
    such as 'hempic(trần ánh dương)'. Force UTF-8 and never fail on printing."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


utf8_console()
