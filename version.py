"""
Version information for Creep Rupture Fitting Tool.

This is the SINGLE SOURCE OF TRUTH for version information.
All other files should import from here.
"""

__version__ = '0.3.0'
__version_info__ = (0, 3, 0)
__release_date__ = '2026-01-21'

def get_version_string():
    """Return formatted version string."""
    return f"v{__version__} ({__release_date__})"

# For compatibility
VERSION = __version__
