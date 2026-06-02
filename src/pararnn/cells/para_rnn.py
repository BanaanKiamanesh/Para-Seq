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
