"""Local-only serving harness for frozen public/offline recommendation artifacts.

This package deliberately is not a production serving system.  It provides a
small, dependency-light integration boundary for replay, contracts and local
operability tests without touching the KAI production flywheel.
"""

from .app import LocalServingApp
from .config import ServingConfig, load_config

__all__ = ["LocalServingApp", "ServingConfig", "load_config"]
