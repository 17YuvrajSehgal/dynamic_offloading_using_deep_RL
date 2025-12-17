"""
Compatibility shim.

The real implementation now lives in `offload_rl.EnvConfig.EnvConfig` so that
the project can be used as a proper Python package. Existing scripts that do
`from EnvConfig import EnvConfig` will continue to work.
"""

from offload_rl.EnvConfig import EnvConfig  # noqa: F401
