"""ferric: Rust-native quantum chemistry. The compiled extension is `ferric.ferric`."""

from . import ferric as _ext
from .ferric import *  # noqa: F403

__doc__ = _ext.__doc__
if hasattr(_ext, "__all__"):
    __all__ = _ext.__all__
