"""Backend dispatch for hardware-specific acceleration.

Fallback chain: cuEquivariance (CUDA) → MLX (Apple Silicon) → vanilla PyTorch (any device).
"""

from foundry import SHOULD_USE_CUEQUIVARIANCE, SHOULD_USE_MLX

__all__ = ["SHOULD_USE_CUEQUIVARIANCE", "SHOULD_USE_MLX"]
