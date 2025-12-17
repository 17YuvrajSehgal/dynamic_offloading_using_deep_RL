"""
Compatibility shim.

`OffloadEnv` now lives in `offload_rl.rl_env`. This module re-exports it so
existing code that imports `OffloadEnv` from `rl_env` keeps working.
"""

from offload_rl.rl_env import OffloadEnv  # noqa: F401
