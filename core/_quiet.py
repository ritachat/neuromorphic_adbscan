"""
_quiet.py  —  Suppress all module print() and warning output.
Import and use:
    from core._quiet import quiet_mode
    with quiet_mode():
        nf.filter(stream)
"""
import contextlib, sys, os, warnings

@contextlib.contextmanager
def quiet_mode():
    """Redirect stdout AND stderr to /dev/null, suppress all warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, 'w') as devnull:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_out
                sys.stderr = old_err
