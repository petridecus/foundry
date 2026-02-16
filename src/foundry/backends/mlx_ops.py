"""MLX implementations of cuEquivariance-equivalent ops for Apple Silicon.

Provides GPU-accelerated triangle attention and triangle multiplicative update
using Apple's MLX framework. These serve as drop-in replacements for the
cuEquivariance kernels when running on M-series chips.

Conversion strategy: torch → numpy → mlx (compute on Apple GPU) → numpy → torch.
"""

import logging
import math

import torch

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    import numpy as np

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def _torch_to_mlx(tensor: torch.Tensor) -> "mx.array":
    """Convert a PyTorch tensor to an MLX array via numpy."""
    return mx.array(tensor.detach().cpu().to(torch.float32).numpy())


def _mlx_to_torch(
    array: "mx.array", device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Convert an MLX array back to a PyTorch tensor."""
    return torch.from_numpy(np.array(array)).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Triangle Attention (replaces cuet.triangle_attention)
# ---------------------------------------------------------------------------


def mlx_triangle_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """MLX triangle attention matching cuEquivariance API.

    Args:
        query: (B, L, H, L, D) — same layout as cuet.triangle_attention input
        key:   (B, L, H, L, D)
        value: (B, L, H, L, D)
        bias:  (B, 1, H, L, L)
        scale: attention scaling factor (1/sqrt(d_head))

    Returns:
        (B, L, H, L, D) — same layout as cuet.triangle_attention output
    """
    orig_device = query.device
    orig_dtype = query.dtype

    q = _torch_to_mlx(query)
    k = _torch_to_mlx(key)
    v = _torch_to_mlx(value)
    b = _torch_to_mlx(bias)

    # Attention scores: (B, L, H, L_q, L_k)
    scores = mx.einsum("blhqd,blhkd->blhqk", q, k) * scale

    # Add bias: (B, 1, H, L, L) broadcasts over L dimension
    scores = scores + b

    # Softmax over key dimension
    weights = mx.softmax(scores, axis=-1)

    # Apply attention: (B, L, H, L_q, D)
    out = mx.einsum("blhqk,blhkd->blhqd", weights, v)

    # Force MLX evaluation before numpy conversion
    mx.eval(out)

    return _mlx_to_torch(out, orig_device, orig_dtype)


# ---------------------------------------------------------------------------
# Triangle Multiplicative Update (replaces cuet.triangle_multiplicative_update)
# ---------------------------------------------------------------------------


def mlx_triangle_multiplicative_update(
    x: torch.Tensor,
    direction: str,
    mask: torch.Tensor | None,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor | None,
    p_in_weight: torch.Tensor,
    g_in_weight: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor | None,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """MLX triangle multiplicative update matching cuEquivariance API.

    This reimplements the fused cuEquivariance kernel as explicit MLX ops.

    Args:
        x: (B, L, L, D) pair representation
        direction: "outgoing" or "incoming"
        mask: unused (kept for API compatibility)
        norm_in_weight, norm_in_bias: LayerNorm parameters
        p_in_weight: (2*D, D) input projection
        g_in_weight: (2*D, D) input gate projection
        norm_out_weight, norm_out_bias: output LayerNorm parameters
        p_out_weight: (D, D) output projection
        g_out_weight: (D, D) output gate projection
        eps: LayerNorm epsilon

    Returns:
        (B, L, L, D) updated pair representation
    """
    orig_device = x.device
    orig_dtype = x.dtype
    D = x.shape[-1]

    # Convert all parameters to MLX
    x_m = _torch_to_mlx(x)
    nin_w = _torch_to_mlx(norm_in_weight)
    nin_b = _torch_to_mlx(norm_in_bias) if norm_in_bias is not None else None
    p_in_w = _torch_to_mlx(p_in_weight)
    g_in_w = _torch_to_mlx(g_in_weight)
    nout_w = _torch_to_mlx(norm_out_weight)
    nout_b = _torch_to_mlx(norm_out_bias) if norm_out_bias is not None else None
    p_out_w = _torch_to_mlx(p_out_weight)
    g_out_w = _torch_to_mlx(g_out_weight)

    # --- Input LayerNorm ---
    mean = mx.mean(x_m, axis=-1, keepdims=True)
    var = mx.var(x_m, axis=-1, keepdims=True)
    x_norm = (x_m - mean) * mx.rsqrt(var + eps) * nin_w
    if nin_b is not None:
        x_norm = x_norm + nin_b

    # --- Input projections (fused linear: weight is (2*D, D), applied as x @ W^T) ---
    p_combined = x_norm @ p_in_w.T  # (B, L, L, 2*D)
    left = p_combined[..., :D]
    right = p_combined[..., D:]

    g_combined = x_norm @ g_in_w.T  # (B, L, L, 2*D)
    left_gate = mx.sigmoid(g_combined[..., :D])
    right_gate = mx.sigmoid(g_combined[..., D:])

    left = left_gate * left
    right = right_gate * right

    # --- Triangle multiplication ---
    L = x_m.shape[1]
    if direction == "outgoing":
        out = mx.einsum("bikd,bjkd->bijd", left, right / float(L))
    else:
        out = mx.einsum("bkid,bkjd->bijd", left, right / float(L))

    # --- Output LayerNorm ---
    mean = mx.mean(out, axis=-1, keepdims=True)
    var = mx.var(out, axis=-1, keepdims=True)
    out = (out - mean) * mx.rsqrt(var + eps) * nout_w
    if nout_b is not None:
        out = out + nout_b

    # --- Output projection + gating ---
    out = out @ p_out_w.T
    gate = mx.sigmoid(x_norm @ g_out_w.T)
    out = gate * out

    mx.eval(out)

    return _mlx_to_torch(out, orig_device, orig_dtype)
