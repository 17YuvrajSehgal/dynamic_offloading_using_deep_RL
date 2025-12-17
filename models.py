"""
Compatibility shim.

The real implementations now live in `offload_rl.models`. This file simply
re-exports them so existing imports like `from models import UE, BaseStation`
continue to work.
"""

from offload_rl.models import (  # noqa: F401
    Task,
    TaskFactory,
    MECServer,
    CloudServer,
    BaseStation,
    UE,
)
