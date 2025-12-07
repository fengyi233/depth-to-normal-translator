"""
D2NT: A High-Performing Depth-to-Normal Translator

This package provides functionality to convert depth maps to surface normal maps.
"""

from .core import depth2normal
from .utils import get_normal_vis, get_normal_vis_reference

__version__ = "0.1.1"
__all__ = [
    "depth2normal",
    "get_normal_vis",
    "get_normal_vis_reference",
]

