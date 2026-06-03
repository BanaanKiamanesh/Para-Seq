from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch


ApplicationMode = Literal["sequential", "deer"]
DeerMode = ApplicationMode
ScanBackend = Literal["torch", "accel_scan"]
StoppingCriterion = Literal["update", "merit"]
InitialGuess = Literal["zero", "f0"]
JacobianBackend = Literal["autograd", "explicit"]
BackwardBackend = Literal["autograd", "adjoint"]


@dataclass
class DeerNewtonConfig:
    """Configuration for the DEER/Newton backend.

    This mirrors the arguments of src.algos.DEER.deer_alg and is the first
    ParaRNN backend used in this repository.

    quasi:
        False -> full DEER / full Newton using dense Jacobians.
        True  -> quasi-DEER using diagonal Jacobians.

    jacobian_backend:
        "autograd":
            Assemble Jacobians inside src.algos.DEER using torch.func.jacrev.

        "explicit":
            Let a structured cell, such as ParaGRU, provide its own explicit
            Jacobian callback to src.algos.DEER.deer_alg.

            For the current ParaGRU implementation, this backend is valid with
            quasi=True because ParaGRU provides diagonal Jacobians.

    backward_backend:
        "autograd":
            Differentiate through the full DEER computation graph.

        "adjoint":
            Do not backpropagate through Newton/DEER iterations. Instead, use a
            ParaRNN-style reverse-time adjoint recurrence. This is currently
            implemented for explicit quasi-DEER ParaGRU only.
    """

    num_iters: int = 4
    tol: Optional[float] = None
    quasi: bool = False
    damping: float = 0.0
    clip_value: Optional[float] = None
    return_trace: bool = False
    scan_backend: ScanBackend = "torch"
    accel_module: str = "warp"
    strict_tol: bool = False
    stopping_criterion: StoppingCriterion = "update"
    initial_guess: InitialGuess = "f0"
    jacobian_backend: JacobianBackend = "autograd"
    backward_backend: BackwardBackend = "autograd"


@dataclass
class ParaRNNConfig:
    """Base configuration for ParaRNN-style recurrent cells.

    Tensor convention is batch-first by default:

        x:      (B, T, input_dim)
        states: (B, T, state_dim)
        output: (B, T, output_dim)

    A single unbatched sequence is also accepted:

        x:      (T, input_dim)
        output: (T, output_dim)

    For GRU/RNN cells, usually:

        state_dim == output_dim

    For LSTM-style cells, usually:

        state_dim = 2 * hidden_dim
        output_dim = hidden_dim

    because the internal recurrent state contains both cell state and hidden
    state, while the user-facing output is only the hidden state.
    """

    input_dim: int
    state_dim: int
    output_dim: Optional[int] = None
    mode: ApplicationMode = "deer"
    batch_first: bool = True
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    deer: DeerNewtonConfig = field(default_factory=DeerNewtonConfig)

    def __post_init__(self) -> None:
        if self.output_dim is None:
            self.output_dim = self.state_dim


# Backward-compatible name from Phase 1.
ParaRNNDeerConfig = ParaRNNConfig

# === Native solver extension fields ===

try:
    DeerNewtonConfig.__annotations__["solver"] = str
    DeerNewtonConfig.__annotations__["sigmasq"] = float
    DeerNewtonConfig.__annotations__["process_noise"] = float
    DeerNewtonConfig.__annotations__["pararnn_deer_kind"] = object
    DeerNewtonConfig.__annotations__["pararnn_elk_kind"] = object
    DeerNewtonConfig.__annotations__["paragru_elk_kind"] = object
    DeerNewtonConfig.__annotations__["paralstm_deer_kind"] = object
    DeerNewtonConfig.__annotations__["paralstm_elk_kind"] = object
except Exception:
    pass

# === End native solver extension fields ===
