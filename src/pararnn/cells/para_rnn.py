from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from src.algos.DEER import deer_alg_batched
from src.pararnn.base_cell import BaseParaRNNCell
from src.pararnn.config import DeerNewtonConfig, ParaRNNConfig


ParaRNNBackend = Literal["autograd", "dense_deer_autograd_torch"]
ParaRNNNonlinearity = Literal["tanh", "relu"]


def make_pararnn_deer_config(
    backend: ParaRNNBackend = "autograd",
    *,
    num_iters: int = 4,
    tol: float | None = None,
    strict_tol: bool = False,
    initial_guess: str = "f0",
    scan_backend: Literal["torch"] = "torch",
) -> DeerNewtonConfig:
    """Construct a full dense DEER/Newton config for vanilla ParaRNN.

    This is the full, dense vanilla RNN case. The recurrent Jacobian is an
    H x H dense matrix at every time step, so this uses full DEER rather than
    quasi/diagonal DEER.
    """
    if backend == "dense_deer_autograd_torch":
        backend = "autograd"

    if backend != "autograd":
        raise ValueError(
            f"Unknown ParaRNN backend {backend!r}. Expected 'autograd'."
        )

    if scan_backend != "torch":
        raise ValueError(
            "Full dense ParaRNN DEER currently supports scan_backend='torch' only."
        )

    cfg = DeerNewtonConfig(
        num_iters=num_iters,
        tol=tol,
        strict_tol=strict_tol,
        stopping_criterion="update",
        initial_guess=initial_guess,  # type: ignore[arg-type]
        quasi=False,
        scan_backend="torch",
        accel_module="warp",
    )
    cfg.jacobian_backend = "explicit"
    cfg.backward_backend = "autograd"
    return cfg


def _apply_nonlinearity(preactivation: torch.Tensor, nonlinearity: str) -> torch.Tensor:
    if nonlinearity == "tanh":
        return torch.tanh(preactivation)
    if nonlinearity == "relu":
        return torch.relu(preactivation)
    raise ValueError("nonlinearity must be either 'tanh' or 'relu'.")


def _nonlinearity_derivative_from_preactivation(
    preactivation: torch.Tensor,
    output: torch.Tensor,
    nonlinearity: str,
) -> torch.Tensor:
    if nonlinearity == "tanh":
        return 1.0 - output * output
    if nonlinearity == "relu":
        return (preactivation > 0.0).to(dtype=preactivation.dtype)
    raise ValueError("nonlinearity must be either 'tanh' or 'relu'.")


def functional_pararnn_recurrence_step(
    state: torch.Tensor,
    driver: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor | None,
    bias_hh: torch.Tensor | None,
    nonlinearity: ParaRNNNonlinearity = "tanh",
) -> torch.Tensor:
    """Full dense vanilla RNN step.

    Equation:
        h_t = phi(x_t W_ih^T + h_{t-1} W_hh^T + b_ih + b_hh)

    where W_hh is dense, not diagonal.
    """
    preactivation = driver @ weight_ih.transpose(-1, -2)
    preactivation = preactivation + state @ weight_hh.transpose(-1, -2)

    if bias_ih is not None:
        preactivation = preactivation + bias_ih
    if bias_hh is not None:
        preactivation = preactivation + bias_hh

    return _apply_nonlinearity(preactivation, nonlinearity)


def functional_pararnn_linearization_dense_from_previous(
    previous_states: torch.Tensor,
    drivers: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor | None,
    bias_hh: torch.Tensor | None,
    nonlinearity: ParaRNNNonlinearity = "tanh",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predicted states and exact dense recurrent Jacobians.

    previous_states: (B, T, H)
    drivers:         (B, T, I)

    Returns:
        predicted_states: (B, T, H)
        jacobians:        (B, T, H, H), with

            jacobians[b,t,j,k] = d h_t[j] / d h_{t-1}[k].

    For the full dense RNN,

        J_t = diag(phi'(u_t)) W_hh.
    """
    if previous_states.ndim != 3:
        raise ValueError(
            "previous_states must have shape (B, T, H), got "
            f"{tuple(previous_states.shape)}."
        )

    if drivers.ndim != 3:
        raise ValueError(
            "drivers must have shape (B, T, input_size), got "
            f"{tuple(drivers.shape)}."
        )

    if previous_states.shape[:2] != drivers.shape[:2]:
        raise ValueError(
            "previous_states and drivers must share batch/time dimensions, got "
            f"{tuple(previous_states.shape)} and {tuple(drivers.shape)}."
        )

    preactivation = drivers @ weight_ih.transpose(-1, -2)
    preactivation = preactivation + previous_states @ weight_hh.transpose(-1, -2)

    if bias_ih is not None:
        preactivation = preactivation + bias_ih
    if bias_hh is not None:
        preactivation = preactivation + bias_hh

    predicted_states = _apply_nonlinearity(preactivation, nonlinearity)
    dphi = _nonlinearity_derivative_from_preactivation(
        preactivation=preactivation,
        output=predicted_states,
        nonlinearity=nonlinearity,
    )

    # J[b,t,j,k] = phi'(u[b,t,j]) * W_hh[j,k]
    jacobians = dphi.unsqueeze(-1) * weight_hh

    return predicted_states, jacobians


class ParaRNNCell(nn.Module):
    """Single-step full dense vanilla ParaRNN cell.

    This mirrors torch.nn.RNNCell at the API level, but is used by ParaRNN's
    DEER backend to expose the exact dense recurrent Jacobian.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: ParaRNNNonlinearity = "tanh",
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError("nonlinearity must be either 'tanh' or 'relu'.")

        factory_kwargs = {"device": device, "dtype": dtype}

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.bias_enabled = bool(bias)
        self.nonlinearity = nonlinearity

        self.weight_ih = nn.Parameter(
            torch.empty(hidden_size, input_size, **factory_kwargs)
        )
        self.weight_hh = nn.Parameter(
            torch.empty(hidden_size, hidden_size, **factory_kwargs)
        )

        if self.bias_enabled:
            self.bias_ih = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
            self.bias_hh = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match torch.nn.RNNCell-style initialization.
        bound = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.uniform_(self.weight_ih, -bound, bound)
        torch.nn.init.uniform_(self.weight_hh, -bound, bound)

        if self.bias_ih is not None:
            torch.nn.init.uniform_(self.bias_ih, -bound, bound)
        if self.bias_hh is not None:
            torch.nn.init.uniform_(self.bias_hh, -bound, bound)

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"bias={self.bias_enabled}, nonlinearity={self.nonlinearity!r}, "
            "variant='dense_vanilla_rnn'"
        )

    def forward(
        self,
        input: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        unbatched = input.ndim == 1

        if input.ndim not in (1, 2):
            raise ValueError(
                "ParaRNNCell input must have shape (input_size,) or "
                f"(batch, input_size), got {tuple(input.shape)}."
            )

        if input.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected input.shape[-1] == {self.input_size}, "
                f"got {input.shape[-1]}."
            )

        input_batched = input.unsqueeze(0) if unbatched else input

        if hx is None:
            hx_batched = torch.zeros(
                input_batched.shape[0],
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            )
        else:
            if hx.ndim == 1:
                hx_batched = hx.unsqueeze(0)
            elif hx.ndim == 2:
                hx_batched = hx
            else:
                raise ValueError(
                    "ParaRNNCell hx must have shape (hidden_size,) or "
                    f"(batch, hidden_size), got {tuple(hx.shape)}."
                )

            expected = (input_batched.shape[0], self.hidden_size)
            if tuple(hx_batched.shape) != expected:
                raise ValueError(
                    f"Expected hx shape {expected}, got {tuple(hx_batched.shape)}."
                )

            hx_batched = hx_batched.to(device=input.device, dtype=input.dtype)

        out = self.recurrence_step(hx_batched, input_batched)

        if unbatched:
            return out.squeeze(0)
        return out

    def recurrence_step(
        self,
        state: torch.Tensor,
        driver: torch.Tensor,
    ) -> torch.Tensor:
        return functional_pararnn_recurrence_step(
            state=state,
            driver=driver,
            weight_ih=self.weight_ih,
            weight_hh=self.weight_hh,
            bias_ih=self.bias_ih,
            bias_hh=self.bias_hh,
            nonlinearity=self.nonlinearity,
        )

    def compute_linearization_dense_from_previous(
        self,
        previous_states: torch.Tensor,
        drivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return functional_pararnn_linearization_dense_from_previous(
            previous_states=previous_states,
            drivers=drivers,
            weight_ih=self.weight_ih,
            weight_hh=self.weight_hh,
            bias_ih=self.bias_ih,
            bias_hh=self.bias_hh,
            nonlinearity=self.nonlinearity,
        )


class ParaRNN(BaseParaRNNCell):
    """Sequence-level full dense vanilla ParaRNN, analogous to torch.nn.RNN.

    Public call:
        output, h_n = rnn(input, hx=None)

    Current implementation scope:
        * one recurrent layer,
        * one direction,
        * full dense recurrent matrix W_hh,
        * tanh or relu nonlinearity,
        * sequential or full-DEER forward modes,
        * pure PyTorch dense associative scan.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: ParaRNNNonlinearity = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        *,
        mode: Literal["sequential", "deer"] = "sequential",
        deer_config: DeerNewtonConfig | None = None,
        backend: ParaRNNBackend = "autograd",
        scan_backend: Literal["torch"] = "torch",
        num_iters: int = 4,
        tol: float | None = None,
        strict_tol: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        if num_layers != 1:
            raise NotImplementedError("ParaRNN currently supports num_layers=1 only.")
        if bidirectional:
            raise NotImplementedError("ParaRNN currently supports bidirectional=False only.")
        if dropout != 0.0:
            raise NotImplementedError("ParaRNN currently supports dropout=0.0 only.")
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError("nonlinearity must be either 'tanh' or 'relu'.")

        if deer_config is None:
            deer_config = make_pararnn_deer_config(
                backend=backend,
                num_iters=num_iters,
                tol=tol,
                strict_tol=strict_tol,
                scan_backend=scan_backend,
            )

        config = ParaRNNConfig(
            input_dim=input_size,
            state_dim=hidden_size,
            output_dim=hidden_size,
            mode=mode,
            batch_first=batch_first,
            device=torch.device(device) if device is not None else None,
            dtype=dtype,
            deer=deer_config,
        )

        super().__init__(config)

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.nonlinearity = nonlinearity
        self.bias = bool(bias)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)

        self.cell = ParaRNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            nonlinearity=nonlinearity,
            device=device,
            dtype=dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, nonlinearity={self.nonlinearity!r}, "
            f"bias={self.bias}, batch_first={self.batch_first}, mode={self.mode}, "
            "variant='dense_vanilla_rnn'"
        )

    @property
    def weight_ih(self) -> torch.nn.Parameter:
        return self.cell.weight_ih

    @property
    def weight_hh(self) -> torch.nn.Parameter:
        return self.cell.weight_hh

    @property
    def bias_ih(self) -> torch.Tensor | None:
        return self.cell.bias_ih

    @property
    def bias_hh(self) -> torch.Tensor | None:
        return self.cell.bias_hh

    def reset_parameters(self) -> None:
        self.cell.reset_parameters()

    def recurrence_step(
        self,
        state: torch.Tensor,
        driver: torch.Tensor,
    ) -> torch.Tensor:
        return self.cell.recurrence_step(state, driver)

    def forward(
        self,
        input: torch.Tensor,
        hx: torch.Tensor | None = None,
        *,
        mode: Literal["sequential", "deer"] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initial_state, unbatched_input = self._hx_to_initial_state(input, hx)
        selected_mode = self.mode if mode is None else mode

        if selected_mode == "sequential":
            output = self.forward_sequential(input, initial_state=initial_state)
        elif selected_mode == "deer":
            output = self.forward_deer(input, initial_state=initial_state)
        else:
            raise ValueError(
                f"Unknown mode {selected_mode!r}. Expected 'sequential' or 'deer'."
            )

        h_n = self._make_h_n(output, unbatched_input=unbatched_input)
        return output, h_n

    def forward_deer(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        deer_config: DeerNewtonConfig | None = None,
    ) -> torch.Tensor:
        cfg = self.config.deer if deer_config is None else deer_config
        self._validate_dense_deer_config(cfg)

        x_batched, had_batch_dim = self._normalize_input(x)
        initial_state_batched = self._normalize_initial_state(
            x_batched=x_batched,
            initial_state=initial_state,
        )

        states_guess = self.assemble_initial_guess_batched(
            drivers=x_batched,
            initial_state=initial_state_batched,
            guess_type=cfg.initial_guess,
        )

        states, info = deer_alg_batched(
            f=self.recurrence_step,
            initial_state=initial_state_batched,
            states_guess=states_guess,
            drivers=x_batched,
            num_iters=cfg.num_iters,
            tol=cfg.tol,
            quasi=False,
            damping=cfg.damping,
            clip_value=cfg.clip_value,
            return_trace=cfg.return_trace,
            scan_backend="torch",
            accel_scan_fn=None,
            strict_tol=cfg.strict_tol,
            stopping_criterion=cfg.stopping_criterion,
            linearization_fn=self.cell.compute_linearization_dense_from_previous,
        )

        info["jacobian_backend"] = "explicit_dense"
        info["linearization_backend"] = "custom_dense"
        info["backward_backend"] = "autograd"
        info["cell_variant"] = f"dense_vanilla_{self.nonlinearity}"
        self.last_deer_infos = [info]

        outputs = self.post_process(states)
        return self._restore_output_layout(outputs, had_batch_dim=had_batch_dim)

    def _hx_to_initial_state(
        self,
        input: torch.Tensor,
        hx: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, bool]:
        if input.ndim == 2:
            unbatched = True
            batch_size = 1
        elif input.ndim == 3:
            unbatched = False
            batch_size = input.shape[0] if self.batch_first else input.shape[1]
        else:
            raise ValueError(
                "ParaRNN input must have shape (L, input_size), "
                "(L, N, input_size), or (N, L, input_size)."
            )

        if hx is None:
            return None, unbatched

        hx = hx.to(device=input.device, dtype=input.dtype)

        if unbatched:
            if hx.ndim == 1:
                if hx.shape != (self.hidden_size,):
                    raise ValueError(
                        f"Expected hx shape ({self.hidden_size},), got {tuple(hx.shape)}."
                    )
                return hx, True
            if hx.ndim == 2:
                if hx.shape != (1, self.hidden_size):
                    raise ValueError(
                        f"Expected hx shape (1, {self.hidden_size}), got {tuple(hx.shape)}."
                    )
                return hx.squeeze(0), True
            raise ValueError(
                "For unbatched input, hx must have shape (hidden_size,) or "
                "(1, hidden_size)."
            )

        if hx.ndim == 2:
            expected = (batch_size, self.hidden_size)
            if tuple(hx.shape) != expected:
                raise ValueError(
                    f"Expected hx shape {expected}, got {tuple(hx.shape)}."
                )
            return hx, False

        if hx.ndim == 3:
            expected = (1, batch_size, self.hidden_size)
            if tuple(hx.shape) != expected:
                raise ValueError(
                    f"Expected hx shape {expected}, got {tuple(hx.shape)}."
                )
            return hx[0], False

        raise ValueError(
            "For batched input, hx must have shape (batch, hidden_size) or "
            "(1, batch, hidden_size)."
        )

    def _make_h_n(
        self,
        output: torch.Tensor,
        *,
        unbatched_input: bool,
    ) -> torch.Tensor:
        if output.shape[-2] == 0:
            raise ValueError("Cannot compute h_n for an empty sequence.")

        if unbatched_input:
            return output[-1, :].unsqueeze(0)

        if self.batch_first:
            return output[:, -1, :].unsqueeze(0)

        return output[-1, :, :].unsqueeze(0)

    @staticmethod
    def _validate_dense_deer_config(cfg: DeerNewtonConfig) -> None:
        if cfg.quasi:
            raise ValueError("Full dense ParaRNN DEER requires quasi=False.")
        if cfg.scan_backend != "torch":
            raise ValueError(
                "Full dense ParaRNN DEER currently supports scan_backend='torch' only."
            )
        if getattr(cfg, "jacobian_backend", "explicit") != "explicit":
            raise ValueError(
                "Full dense ParaRNN DEER requires jacobian_backend='explicit'."
            )
        if getattr(cfg, "backward_backend", "autograd") != "autograd":
            raise ValueError(
                "Full dense ParaRNN DEER currently supports backward_backend='autograd' only."
            )
        if cfg.stopping_criterion not in ("update", "merit"):
            raise ValueError("stopping_criterion must be 'update' or 'merit'.")


# === Dense ParaRNN scalar quasi-DEER extension ===
#
# Existing dense ParaRNN full-DEER uses the exact dense Jacobian
#
#     J_t = diag(phi'(u_t)) @ W_hh.
#
# This extension adds approximate scalar quasi-DEER by keeping only
#
#     diag(J_t) = phi'(u_t) * diag(W_hh).
#
# Therefore the Newton linear solve becomes a diagonal affine recurrence and can
# use either torch associative_scan or accelerated_scan.

def functional_pararnn_linearization_diag_from_previous(
    previous_states: torch.Tensor,
    drivers: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor | None,
    bias_hh: torch.Tensor | None,
    nonlinearity: ParaRNNNonlinearity = "tanh",
) -> tuple[torch.Tensor, torch.Tensor]:
    if previous_states.ndim != 3:
        raise ValueError(
            "previous_states must have shape (B, T, H), got "
            f"{tuple(previous_states.shape)}."
        )

    if drivers.ndim != 3:
        raise ValueError(
            "drivers must have shape (B, T, input_size), got "
            f"{tuple(drivers.shape)}."
        )

    if previous_states.shape[:2] != drivers.shape[:2]:
        raise ValueError(
            "previous_states and drivers must share batch/time dimensions, got "
            f"{tuple(previous_states.shape)} and {tuple(drivers.shape)}."
        )

    preactivation = drivers @ weight_ih.transpose(-1, -2)
    preactivation = preactivation + previous_states @ weight_hh.transpose(-1, -2)

    if bias_ih is not None:
        preactivation = preactivation + bias_ih

    if bias_hh is not None:
        preactivation = preactivation + bias_hh

    predicted_states = _apply_nonlinearity(preactivation, nonlinearity)

    dphi = _nonlinearity_derivative_from_preactivation(
        preactivation=preactivation,
        output=predicted_states,
        nonlinearity=nonlinearity,
    )

    recurrent_diag = torch.diagonal(weight_hh, dim1=-2, dim2=-1)
    jacobian_diag = dphi * recurrent_diag

    return predicted_states, jacobian_diag


def _pararnn_cell_compute_linearization_diag_from_previous(
    self,
    previous_states: torch.Tensor,
    drivers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return functional_pararnn_linearization_diag_from_previous(
        previous_states=previous_states,
        drivers=drivers,
        weight_ih=self.weight_ih,
        weight_hh=self.weight_hh,
        bias_ih=self.bias_ih,
        bias_hh=self.bias_hh,
        nonlinearity=self.nonlinearity,
    )


def _pararnn_compute_linearization_diag_from_previous(
    self,
    previous_states: torch.Tensor,
    drivers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return self.cell.compute_linearization_diag_from_previous(
        previous_states=previous_states,
        drivers=drivers,
    )


def _pararnn_compute_jacobians_diag_from_previous(
    self,
    previous_states: torch.Tensor,
    drivers: torch.Tensor,
) -> torch.Tensor:
    _, jacobian_diag = self.compute_linearization_diag_from_previous(
        previous_states=previous_states,
        drivers=drivers,
    )
    return jacobian_diag


ParaRNNCell.compute_linearization_diag_from_previous = (
    _pararnn_cell_compute_linearization_diag_from_previous
)

ParaRNN.compute_linearization_diag_from_previous = (
    _pararnn_compute_linearization_diag_from_previous
)

ParaRNN._compute_jacobians_diag_from_previous = (
    _pararnn_compute_jacobians_diag_from_previous
)


def make_pararnn_deer_config(
    backend: str = "autograd",
    *,
    num_iters: int = 4,
    tol: float | None = None,
    strict_tol: bool = False,
    initial_guess: str = "f0",
    scan_backend: str = "torch",
    accel_module: str = "warp",
) -> DeerNewtonConfig:
    """Construct a DEER config for dense vanilla ParaRNN.

    Backends:
        autograd / dense_deer_autograd_torch:
            exact dense full-DEER with torch dense associative scan.

        quasi / quasi_autograd / quasi_deer_autograd_torch:
            approximate scalar quasi-DEER with torch diagonal scan.

        quasi_deer_autograd_accel_scan:
            approximate scalar quasi-DEER with accelerated_scan.
    """
    if backend in ("dense_deer_autograd_torch", "full", "full_deer_autograd_torch"):
        backend = "autograd"
        scan_backend = "torch"

    if backend == "quasi":
        backend = "quasi_autograd"

    if backend == "quasi_deer_autograd_torch":
        backend = "quasi_autograd"
        scan_backend = "torch"

    if backend == "quasi_deer_autograd_accel_scan":
        backend = "quasi_autograd"
        scan_backend = "accel_scan"

    if backend == "autograd":
        if scan_backend != "torch":
            raise ValueError(
                "Full dense ParaRNN DEER supports scan_backend='torch' only. "
                "Use backend='quasi_autograd' for scan_backend='accel_scan'."
            )

        cfg = DeerNewtonConfig(
            num_iters=num_iters,
            tol=tol,
            strict_tol=strict_tol,
            stopping_criterion="update",
            initial_guess=initial_guess,  # type: ignore[arg-type]
            quasi=False,
            scan_backend="torch",
            accel_module=accel_module,
        )
        cfg.jacobian_backend = "explicit"
        cfg.backward_backend = "autograd"
        cfg.pararnn_deer_kind = "full_dense"
        return cfg

    if backend == "quasi_autograd":
        if scan_backend not in ("torch", "accel_scan"):
            raise ValueError(
                "ParaRNN quasi-DEER scan_backend must be 'torch' or 'accel_scan'."
            )

        cfg = DeerNewtonConfig(
            num_iters=num_iters,
            tol=tol,
            strict_tol=strict_tol,
            stopping_criterion="update",
            initial_guess=initial_guess,  # type: ignore[arg-type]
            quasi=True,
            scan_backend=scan_backend,
            accel_module=accel_module,
        )
        cfg.jacobian_backend = "explicit"
        cfg.backward_backend = "autograd"
        cfg.pararnn_deer_kind = "scalar_quasi"
        return cfg

    raise ValueError(
        f"Unknown ParaRNN backend {backend!r}. Expected 'autograd', "
        "'dense_deer_autograd_torch', 'quasi', 'quasi_autograd', "
        "'quasi_deer_autograd_torch', or 'quasi_deer_autograd_accel_scan'."
    )


_ParaRNN_full_dense_forward_deer = ParaRNN.forward_deer


def _pararnn_forward_deer_with_quasi(
    self,
    x: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    deer_config: DeerNewtonConfig | None = None,
) -> torch.Tensor:
    cfg = self.config.deer if deer_config is None else deer_config

    if not getattr(cfg, "quasi", False):
        return _ParaRNN_full_dense_forward_deer(
            self,
            x=x,
            initial_state=initial_state,
            deer_config=cfg,
        )

    if cfg.scan_backend not in ("torch", "accel_scan"):
        raise ValueError(
            "ParaRNN quasi-DEER scan_backend must be 'torch' or 'accel_scan'."
        )

    if getattr(cfg, "jacobian_backend", "explicit") != "explicit":
        raise ValueError("ParaRNN quasi-DEER requires jacobian_backend='explicit'.")

    if getattr(cfg, "backward_backend", "autograd") != "autograd":
        raise ValueError(
            "ParaRNN quasi-DEER currently supports backward_backend='autograd' only."
        )

    if cfg.return_trace:
        raise ValueError("ParaRNN quasi-DEER does not support return_trace=True yet.")

    x_batched, had_batch_dim = self._normalize_input(x)

    initial_state_batched = self._normalize_initial_state(
        x_batched=x_batched,
        initial_state=initial_state,
    )

    states_guess = self.assemble_initial_guess_batched(
        drivers=x_batched,
        initial_state=initial_state_batched,
        guess_type=cfg.initial_guess,
    )

    accel_scan_fn = self._load_accel_scan_if_needed(cfg)

    states, info = deer_alg_batched(
        f=self.recurrence_step,
        initial_state=initial_state_batched,
        states_guess=states_guess,
        drivers=x_batched,
        num_iters=cfg.num_iters,
        tol=cfg.tol,
        quasi=True,
        damping=cfg.damping,
        clip_value=cfg.clip_value,
        return_trace=False,
        scan_backend=cfg.scan_backend,
        accel_scan_fn=accel_scan_fn,
        strict_tol=cfg.strict_tol,
        stopping_criterion=cfg.stopping_criterion,
        linearization_fn=self.compute_linearization_diag_from_previous,
    )

    info = dict(info)
    info["jacobian_backend"] = "explicit_diag_from_dense"
    info["linearization_backend"] = "custom_diag_from_dense"
    info["backward_backend"] = "autograd"
    info["cell_variant"] = f"dense_vanilla_{self.nonlinearity}_scalar_quasi"
    info["pararnn_deer_kind"] = "scalar_quasi"

    self.last_deer_infos = [info]

    outputs = self.post_process(states)

    return self._restore_output_layout(
        outputs,
        had_batch_dim=had_batch_dim,
    )


ParaRNN.forward_deer = _pararnn_forward_deer_with_quasi

try:
    ParaRNNBackend = Literal[
        "autograd",
        "dense_deer_autograd_torch",
        "full",
        "full_deer_autograd_torch",
        "quasi",
        "quasi_autograd",
        "quasi_deer_autograd_torch",
        "quasi_deer_autograd_accel_scan",
    ]
except Exception:
    pass

# === End dense ParaRNN scalar quasi-DEER extension ===


# === Native ParaRNN ELK helpers ===

from src.algos.ELK import elk_alg_batched


def make_pararnn_elk_config(
    backend: str = "elk",
    *,
    num_iters: int = 8,
    tol: float | None = None,
    strict_tol: bool = False,
    initial_guess: str = "f0",
    scan_backend: str = "torch",
    accel_module: str = "warp",
    sigmasq: float = 1e8,
    process_noise: float = 1.0,
) -> DeerNewtonConfig:
    if backend in ("elk", "full_elk", "dense_elk", "autograd"):
        if scan_backend != "torch":
            raise ValueError("Full dense ParaRNN ELK supports scan_backend='torch' only.")

        cfg = DeerNewtonConfig(
            num_iters=num_iters,
            tol=tol,
            strict_tol=strict_tol,
            stopping_criterion="update",
            initial_guess=initial_guess,  # type: ignore[arg-type]
            quasi=False,
            scan_backend="torch",
            accel_module=accel_module,
            jacobian_backend="explicit",
            backward_backend="autograd",
        )
        cfg.solver = "elk"
        cfg.sigmasq = float(sigmasq)
        cfg.process_noise = float(process_noise)
        cfg.pararnn_elk_kind = "full_dense"
        return cfg

    if backend in ("quasi_elk", "scalar_quasi_elk", "quasi", "quasi_autograd"):
        if scan_backend not in ("torch", "accel_scan"):
            raise ValueError("ParaRNN quasi-ELK scan_backend must be 'torch' or 'accel_scan'.")

        cfg = DeerNewtonConfig(
            num_iters=num_iters,
            tol=tol,
            strict_tol=strict_tol,
            stopping_criterion="update",
            initial_guess=initial_guess,  # type: ignore[arg-type]
            quasi=True,
            scan_backend=scan_backend,
            accel_module=accel_module,
            jacobian_backend="explicit",
            backward_backend="autograd",
        )
        cfg.solver = "elk"
        cfg.sigmasq = float(sigmasq)
        cfg.process_noise = float(process_noise)
        cfg.pararnn_elk_kind = "scalar_quasi"
        return cfg

    raise ValueError(
        f"Unknown ParaRNN ELK backend {backend!r}. Expected 'elk', "
        "'full_elk', 'dense_elk', 'quasi_elk', or 'scalar_quasi_elk'."
    )


def _native_pararnn_forward_elk(
    self,
    x: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    elk_config: DeerNewtonConfig | None = None,
) -> torch.Tensor:
    cfg = self.config.deer if elk_config is None else elk_config

    x_batched, had_batch_dim = self._normalize_input(x)
    initial_state_batched = self._normalize_initial_state(
        x_batched=x_batched,
        initial_state=initial_state,
    )

    states_guess = self.assemble_initial_guess_batched(
        drivers=x_batched,
        initial_state=initial_state_batched,
        guess_type=cfg.initial_guess,
    )

    accel_scan_fn = self._load_accel_scan_if_needed(cfg)

    linearization_fn = (
        self.compute_linearization_diag_from_previous
        if cfg.quasi
        else self.cell.compute_linearization_dense_from_previous
    )

    states, info = elk_alg_batched(
        f=self.recurrence_step,
        initial_state=initial_state_batched,
        states_guess=states_guess,
        drivers=x_batched,
        sigmasq=getattr(cfg, "sigmasq", 1e8),
        process_noise=getattr(cfg, "process_noise", 1.0),
        num_iters=cfg.num_iters,
        tol=cfg.tol,
        quasi=cfg.quasi,
        damping=cfg.damping,
        clip_value=cfg.clip_value,
        return_trace=cfg.return_trace,
        scan_backend=cfg.scan_backend,
        accel_scan_fn=accel_scan_fn,
        strict_tol=cfg.strict_tol,
        stopping_criterion=cfg.stopping_criterion,
        linearization_fn=linearization_fn,
    )

    info = dict(info)
    info["solver"] = "elk"
    info["jacobian_backend"] = "explicit_diag_from_dense" if cfg.quasi else "explicit_dense"
    info["linearization_backend"] = "custom_diag_from_dense" if cfg.quasi else "custom_dense"
    info["backward_backend"] = "autograd"
    info["cell_variant"] = (
        f"dense_vanilla_{self.nonlinearity}_quasi_elk"
        if cfg.quasi
        else f"dense_vanilla_{self.nonlinearity}_full_elk"
    )
    info["pararnn_elk_kind"] = getattr(cfg, "pararnn_elk_kind", None)
    self.last_deer_infos = [info]

    outputs = self.post_process(states)

    return self._restore_output_layout(
        outputs,
        had_batch_dim=had_batch_dim,
    )


ParaRNN.forward_elk = _native_pararnn_forward_elk

# === End native ParaRNN ELK helpers ===


# === Part-5 fix: native ELK/adjoint compatibility ===

from src.algos.ELK import elk_alg_batched as _part5_elk_alg_batched


def _part5_make_pararnn_elk_config(
    backend: str = "elk",
    *,
    num_iters: int = 8,
    tol: float | None = None,
    strict_tol: bool = False,
    initial_guess: str = "f0",
    scan_backend: str = "torch",
    accel_module: str = "warp",
    sigmasq: float = 1e8,
    process_noise: float = 1.0,
) -> DeerNewtonConfig:
    if backend in ("elk", "full_elk", "dense_elk", "autograd"):
        if scan_backend != "torch":
            raise ValueError("Full dense ParaRNN ELK supports scan_backend='torch' only.")

        cfg = DeerNewtonConfig(
            num_iters=num_iters,
            tol=tol,
            strict_tol=strict_tol,
            stopping_criterion="update",
            initial_guess=initial_guess,  # type: ignore[arg-type]
            quasi=False,
            scan_backend="torch",
            accel_module=accel_module,
            jacobian_backend="explicit",
            backward_backend="autograd",
        )
        cfg.solver = "elk"
        cfg.sigmasq = float(sigmasq)
        cfg.process_noise = float(process_noise)
        cfg.pararnn_elk_kind = "full_dense"
        return cfg

    if backend in ("quasi_elk", "scalar_quasi_elk", "quasi", "quasi_autograd"):
        if scan_backend not in ("torch", "accel_scan"):
            raise ValueError("ParaRNN quasi-ELK scan_backend must be 'torch' or 'accel_scan'.")

        cfg = DeerNewtonConfig(
            num_iters=num_iters,
            tol=tol,
            strict_tol=strict_tol,
            stopping_criterion="update",
            initial_guess=initial_guess,  # type: ignore[arg-type]
            quasi=True,
            scan_backend=scan_backend,
            accel_module=accel_module,
            jacobian_backend="explicit",
            backward_backend="autograd",
        )
        cfg.solver = "elk"
        cfg.sigmasq = float(sigmasq)
        cfg.process_noise = float(process_noise)
        cfg.pararnn_elk_kind = "scalar_quasi"
        return cfg

    raise ValueError(
        f"Unknown ParaRNN ELK backend {backend!r}. Expected 'elk', "
        "'full_elk', 'dense_elk', 'quasi_elk', or 'scalar_quasi_elk'."
    )


make_pararnn_elk_config = _part5_make_pararnn_elk_config


if not hasattr(ParaRNN, "_part5_original_init"):
    ParaRNN._part5_original_init = ParaRNN.__init__


def _part5_pararnn_init(
    self,
    *args,
    solver: str = "deer",
    elk_sigmasq: float = 1e8,
    elk_process_noise: float = 1.0,
    **kwargs,
):
    if solver in ("elk", "full_elk", "dense_elk", "quasi_elk", "scalar_quasi_elk"):
        requested_mode = kwargs.get("mode", None)
        if requested_mode is None or requested_mode == "deer":
            kwargs["mode"] = "elk"

        if solver in ("quasi_elk", "scalar_quasi_elk"):
            elk_backend = "quasi_elk"
        else:
            elk_backend = kwargs.get("backend", "elk")
            if elk_backend not in ("quasi_elk", "scalar_quasi_elk", "quasi", "quasi_autograd"):
                elk_backend = "elk"

        if kwargs.get("deer_config", None) is None:
            kwargs["deer_config"] = make_pararnn_elk_config(
                backend=elk_backend,
                num_iters=kwargs.get("num_iters", 8),
                tol=kwargs.get("tol", None),
                strict_tol=kwargs.get("strict_tol", False),
                scan_backend=kwargs.get("scan_backend", "torch"),
                accel_module=kwargs.get("accel_module", "warp"),
                sigmasq=elk_sigmasq,
                process_noise=elk_process_noise,
            )

        ParaRNN._part5_original_init(self, *args, **kwargs)
        self.solver = "elk"
        return

    ParaRNN._part5_original_init(self, *args, **kwargs)
    self.solver = getattr(self.config.deer, "solver", "deer")


def _part5_pararnn_forward_elk(
    self,
    x: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    elk_config: DeerNewtonConfig | None = None,
) -> torch.Tensor:
    cfg = self.config.deer if elk_config is None else elk_config

    x_batched, had_batch_dim = self._normalize_input(x)
    initial_state_batched = self._normalize_initial_state(
        x_batched=x_batched,
        initial_state=initial_state,
    )

    states_guess = self.assemble_initial_guess_batched(
        drivers=x_batched,
        initial_state=initial_state_batched,
        guess_type=cfg.initial_guess,
    )

    accel_scan_fn = self._load_accel_scan_if_needed(cfg)

    if cfg.quasi:
        linearization_fn = self.compute_linearization_diag_from_previous
    else:
        linearization_fn = self.cell.compute_linearization_dense_from_previous

    states, info = _part5_elk_alg_batched(
        f=self.recurrence_step,
        initial_state=initial_state_batched,
        states_guess=states_guess,
        drivers=x_batched,
        sigmasq=getattr(cfg, "sigmasq", 1e8),
        process_noise=getattr(cfg, "process_noise", 1.0),
        num_iters=cfg.num_iters,
        tol=cfg.tol,
        quasi=cfg.quasi,
        damping=cfg.damping,
        clip_value=cfg.clip_value,
        return_trace=cfg.return_trace,
        scan_backend=cfg.scan_backend,
        accel_scan_fn=accel_scan_fn,
        strict_tol=cfg.strict_tol,
        stopping_criterion=cfg.stopping_criterion,
        linearization_fn=linearization_fn,
    )

    info = dict(info)
    info["solver"] = "elk"
    info["jacobian_backend"] = "explicit_diag_from_dense" if cfg.quasi else "explicit_dense"
    info["linearization_backend"] = "custom_diag_from_dense" if cfg.quasi else "custom_dense"
    info["backward_backend"] = "autograd"
    info["cell_variant"] = (
        f"dense_vanilla_{self.nonlinearity}_quasi_elk"
        if cfg.quasi
        else f"dense_vanilla_{self.nonlinearity}_full_elk"
    )
    info["pararnn_elk_kind"] = getattr(cfg, "pararnn_elk_kind", None)
    self.last_deer_infos = [info]

    outputs = self.post_process(states)
    return self._restore_output_layout(outputs, had_batch_dim=had_batch_dim)


def _part5_pararnn_forward(
    self,
    input: torch.Tensor,
    hx: torch.Tensor | None = None,
    *,
    mode=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_state, unbatched_input = self._hx_to_initial_state(input, hx)
    selected_mode = self.mode if mode is None else mode

    if selected_mode == "sequential":
        output = self.forward_sequential(input, initial_state=initial_state)
    elif selected_mode == "deer":
        output = self.forward_deer(input, initial_state=initial_state)
    elif selected_mode == "elk":
        output = self.forward_elk(input, initial_state=initial_state)
    else:
        raise ValueError(
            f"Unknown mode {selected_mode!r}. "
            "Expected 'sequential', 'deer', or 'elk'."
        )

    h_n = self._make_h_n(output, unbatched_input=unbatched_input)
    return output, h_n


ParaRNN.__init__ = _part5_pararnn_init
ParaRNN.forward_elk = _part5_pararnn_forward_elk
ParaRNN.forward = _part5_pararnn_forward

# === End Part-5 fix: native ELK/adjoint compatibility ===


# === Unified algorithm mode support ===

def _all_algos_make_fixed_config(
    solver: str,
    *,
    num_iters: int = 20,
    tol: float | None = None,
    strict_tol: bool = False,
    initial_guess: str = "f0",
) -> DeerNewtonConfig:
    if solver not in ("jacobi", "picard"):
        raise ValueError("solver must be 'jacobi' or 'picard'.")

    cfg = DeerNewtonConfig(
        num_iters=num_iters,
        tol=tol,
        strict_tol=strict_tol,
        stopping_criterion="merit",
        initial_guess=initial_guess,  # type: ignore[arg-type]
        quasi=False,
        scan_backend="torch",
        accel_module="warp",
    )
    cfg.solver = solver
    cfg.jacobian_backend = "none"
    cfg.backward_backend = "autograd"
    return cfg


def make_pararnn_jacobi_config(
    *,
    num_iters: int = 20,
    tol: float | None = None,
    strict_tol: bool = False,
    initial_guess: str = "f0",
) -> DeerNewtonConfig:
    return _all_algos_make_fixed_config(
        "jacobi",
        num_iters=num_iters,
        tol=tol,
        strict_tol=strict_tol,
        initial_guess=initial_guess,
    )


def make_pararnn_picard_config(
    *,
    num_iters: int = 20,
    tol: float | None = None,
    strict_tol: bool = False,
    initial_guess: str = "f0",
) -> DeerNewtonConfig:
    return _all_algos_make_fixed_config(
        "picard",
        num_iters=num_iters,
        tol=tol,
        strict_tol=strict_tol,
        initial_guess=initial_guess,
    )


if not hasattr(ParaRNN, "_all_algos_previous_init"):
    ParaRNN._all_algos_previous_init = ParaRNN.__init__


def _all_algos_pararnn_init(
    self,
    *args,
    solver: str = "deer",
    **kwargs,
):
    if solver in ("jacobi", "picard"):
        if kwargs.get("deer_config", None) is None:
            if solver == "jacobi":
                kwargs["deer_config"] = make_pararnn_jacobi_config(
                    num_iters=kwargs.get("num_iters", 20),
                    tol=kwargs.get("tol", None),
                    strict_tol=kwargs.get("strict_tol", False),
                )
            else:
                kwargs["deer_config"] = make_pararnn_picard_config(
                    num_iters=kwargs.get("num_iters", 20),
                    tol=kwargs.get("tol", None),
                    strict_tol=kwargs.get("strict_tol", False),
                )

        if "mode" not in kwargs:
            kwargs["mode"] = solver

        ParaRNN._all_algos_previous_init(self, *args, solver="deer", **kwargs)
        self.solver = solver
        self.mode = kwargs["mode"]
        self.config.mode = self.mode
        return

    ParaRNN._all_algos_previous_init(self, *args, solver=solver, **kwargs)


if not hasattr(ParaRNN, "_all_algos_previous_forward"):
    ParaRNN._all_algos_previous_forward = ParaRNN.forward


def _all_algos_pararnn_forward(
    self,
    input: torch.Tensor,
    hx: torch.Tensor | None = None,
    *,
    mode=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_mode = self.mode if mode is None else mode

    if selected_mode not in ("jacobi", "picard"):
        return ParaRNN._all_algos_previous_forward(
            self,
            input,
            hx,
            mode=mode,
        )

    initial_state, unbatched_input = self._hx_to_initial_state(input, hx)

    if selected_mode == "jacobi":
        output = self.forward_jacobi(input, initial_state=initial_state)
    else:
        output = self.forward_picard(input, initial_state=initial_state)

    h_n = self._make_h_n(output, unbatched_input=unbatched_input)
    return output, h_n


ParaRNN.__init__ = _all_algos_pararnn_init
ParaRNN.forward = _all_algos_pararnn_forward

# === End unified algorithm mode support ===

# === Fixed-point accel_scan support ===

def _fp_accel_make_pararnn_fixed_config(
    solver: str,
    *,
    num_iters: int = 20,
    tol: float | None = None,
    strict_tol: bool = False,
    initial_guess: str = "f0",
    scan_backend: Literal["torch", "accel_scan"] = "torch",
    accel_module: str = "warp",
) -> DeerNewtonConfig:
    if solver not in ("jacobi", "picard"):
        raise ValueError("solver must be 'jacobi' or 'picard'.")
    if scan_backend not in ("torch", "accel_scan"):
        raise ValueError("scan_backend must be 'torch' or 'accel_scan'.")

    cfg = DeerNewtonConfig(
        num_iters=num_iters,
        tol=tol,
        strict_tol=strict_tol,
        stopping_criterion="merit",
        initial_guess=initial_guess,  # type: ignore[arg-type]
        quasi=False,
        scan_backend=scan_backend,
        accel_module=accel_module,
    )
    cfg.solver = solver
    cfg.jacobian_backend = "none"
    cfg.backward_backend = "autograd"
    return cfg


def make_pararnn_jacobi_config(
    *,
    num_iters: int = 20,
    tol: float | None = None,
    strict_tol: bool = False,
    initial_guess: str = "f0",
    scan_backend: Literal["torch", "accel_scan"] = "torch",
    accel_module: str = "warp",
) -> DeerNewtonConfig:
    return _fp_accel_make_pararnn_fixed_config(
        "jacobi",
        num_iters=num_iters,
        tol=tol,
        strict_tol=strict_tol,
        initial_guess=initial_guess,
        scan_backend=scan_backend,
        accel_module=accel_module,
    )


def make_pararnn_picard_config(
    *,
    num_iters: int = 20,
    tol: float | None = None,
    strict_tol: bool = False,
    initial_guess: str = "f0",
    scan_backend: Literal["torch", "accel_scan"] = "torch",
    accel_module: str = "warp",
) -> DeerNewtonConfig:
    return _fp_accel_make_pararnn_fixed_config(
        "picard",
        num_iters=num_iters,
        tol=tol,
        strict_tol=strict_tol,
        initial_guess=initial_guess,
        scan_backend=scan_backend,
        accel_module=accel_module,
    )


if not hasattr(ParaRNN, "_fp_accel_previous_init"):
    ParaRNN._fp_accel_previous_init = ParaRNN.__init__


def _fp_accel_pararnn_init(
    self,
    *args,
    solver: str = "deer",
    **kwargs,
):
    if solver in ("jacobi", "picard"):
        if kwargs.get("deer_config", None) is None:
            helper = make_pararnn_jacobi_config if solver == "jacobi" else make_pararnn_picard_config
            kwargs["deer_config"] = helper(
                num_iters=kwargs.get("num_iters", 20),
                tol=kwargs.get("tol", None),
                strict_tol=kwargs.get("strict_tol", False),
                scan_backend=kwargs.get("scan_backend", "torch"),
                accel_module=kwargs.get("accel_module", "warp"),
            )

        if "mode" not in kwargs:
            kwargs["mode"] = solver

        ParaRNN._fp_accel_previous_init(self, *args, solver="deer", **kwargs)
        self.solver = solver
        self.mode = kwargs["mode"]
        self.config.mode = self.mode
        return

    ParaRNN._fp_accel_previous_init(self, *args, solver=solver, **kwargs)


ParaRNN.__init__ = _fp_accel_pararnn_init

# === End fixed-point accel_scan support ===


# === Accel fixed-point constructor and public forward final fix ===

if not hasattr(ParaRNN, "_accel_final_original_init"):
    if hasattr(ParaRNN, "_part5_original_init"):
        ParaRNN._accel_final_original_init = ParaRNN._part5_original_init
    else:
        ParaRNN._accel_final_original_init = ParaRNN.__init__


if not hasattr(ParaRNN, "_accel_final_delegate_init"):
    ParaRNN._accel_final_delegate_init = ParaRNN.__init__


def _accel_final_pararnn_init(
    self,
    *args,
    solver: str = "deer",
    **kwargs,
):
    if solver in ("jacobi", "picard"):
        accel_module = kwargs.pop("accel_module", "warp")
        scan_backend = kwargs.get("scan_backend", "torch")

        if kwargs.get("deer_config", None) is None:
            helper = (
                make_pararnn_jacobi_config
                if solver == "jacobi"
                else make_pararnn_picard_config
            )
            kwargs["deer_config"] = helper(
                num_iters=kwargs.get("num_iters", 20),
                tol=kwargs.get("tol", None),
                strict_tol=kwargs.get("strict_tol", False),
                initial_guess=kwargs.get("initial_guess", "f0"),
                scan_backend=scan_backend,
                accel_module=accel_module,
            )

        if "mode" not in kwargs:
            kwargs["mode"] = solver

        ParaRNN._accel_final_original_init(self, *args, **kwargs)

        self.solver = solver
        self.mode = kwargs["mode"]
        self.config.mode = self.mode
        return

    # ParaRNN's older native constructor does not accept accel_module.
    # Preserve the value by requiring callers to pass it through config helpers;
    # avoid leaking it into the original constructor chain.
    if "accel_module" in kwargs and kwargs.get("deer_config", None) is not None:
        kwargs.pop("accel_module", None)

    ParaRNN._accel_final_delegate_init(self, *args, solver=solver, **kwargs)


def _accel_final_pararnn_forward(
    self,
    input: torch.Tensor,
    hx: torch.Tensor | None = None,
    *,
    mode=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_state, unbatched_input = self._hx_to_initial_state(input, hx)
    selected_mode = self.mode if mode is None else mode

    if selected_mode == "sequential":
        output = self.forward_sequential(input, initial_state=initial_state)
    elif selected_mode == "deer":
        output = self.forward_deer(input, initial_state=initial_state)
    elif selected_mode == "elk":
        output = self.forward_elk(input, initial_state=initial_state)
    elif selected_mode == "jacobi":
        output = self.forward_jacobi(input, initial_state=initial_state)
    elif selected_mode == "picard":
        output = self.forward_picard(input, initial_state=initial_state)
    else:
        raise ValueError(
            f"Unknown mode {selected_mode!r}. "
            "Expected 'sequential', 'deer', 'elk', 'jacobi', or 'picard'."
        )

    h_n = self._make_h_n(output, unbatched_input=unbatched_input)
    return output, h_n


ParaRNN.__init__ = _accel_final_pararnn_init
ParaRNN.forward = _accel_final_pararnn_forward

# === End accel fixed-point constructor and public forward final fix ===
