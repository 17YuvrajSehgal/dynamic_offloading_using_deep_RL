"""
Compatibility shim.

`ActorCriticAgent` and its networks now live in `offload_rl.ac_agent`. This
module re-exports them so existing imports keep working.
"""

from offload_rl.ac_agent import (  # noqa: F401
    ActorNet,
    CriticNet,
    ActorCriticAgent,
)
