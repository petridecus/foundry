"""Diffusion step callback protocol for streaming intermediate results."""

from dataclasses import dataclass

import torch


@dataclass
class StepInfo:
    """Information about a single diffusion sampling step.

    Passed to step_callback functions during inference sampling.

    Attributes:
        step: 1-indexed step number.
        total_steps: Total number of diffusion steps.
        coords: Denoised coordinates tensor, shape (D, L, 3). Stays on-device;
            consumer is responsible for calling .cpu() if needed.
        noise_level: Current t_hat value (scalar).
        sequence_logits: Per-residue sequence logits, shape (D, L, 32). RFD3 only.
        motif_mask: Boolean mask of fixed motif atoms, shape (L,). RFD3 only.
    """

    step: int
    total_steps: int
    coords: torch.Tensor
    noise_level: float
    sequence_logits: torch.Tensor | None = None
    motif_mask: torch.Tensor | None = None
