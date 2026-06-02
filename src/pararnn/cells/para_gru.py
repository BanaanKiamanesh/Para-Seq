from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from src.algos.DEER import merit_fxn_batched
from src.pararnn.adjoint import (
    functional_paragru_input_projection,
    functional_paragru_linearization_diag_from_previous,
    functional_paragru_recurrence_step,
    paragru_adjoint_deer_forward,
)
from src.pararnn.base_cell import BaseParaRNNCell
from src.pararnn.config import DeerNewtonConfig, ParaRNNConfig


ParaGRUBackend = Literal[
    "autograd",
    "adjoint",
    "deer_autograd_torch",
    "deer_adjoint_torch",
    "deer_adjoint_accel_scan",
]


@dataclass
class ParaGRUConfig(ParaRNNConfig):
    """Configuration used internally by the sequence-level ParaGRU.

    The public API is intentionally PyTorch-like and does not require users to
    instantiate this dataclass directly. It remains available for low-level and
    backward-compatible tests.
    """

    recurrent_init_scale: float = 0.25
    input_init_scale: float = 1.0
    bias_init_value: float = 0.0
    bias: bool = True


def make_paragru_deer_config(
    backend: ParaGRUBackend = "adjoint",
    *,
    num_iters: int = 4,
    tol: float | None = None,
    strict_tol: bool = False,
    initial_guess: str = "f0",
    scan_backend: Literal["torch", "accel_scan"] = "torch",
    accel_module: str = "warp",
) -> DeerNewtonConfig:
    """Construct a DEER config for ParaGRU.

    Args:
        backend:
            "autograd" differentiates through DEER iterations.
            "adjoint" uses the custom ParaGRU adjoint backward.
            The longer legacy names are also accepted.
    """
    if backend == "deer_autograd_torch":
        backend = "autograd"
        scan_backend = "torch"
    elif backend == "deer_adjoint_torch":
        backend = "adjoint"
        scan_backend = "torch"
    elif backend == "deer_adjoint_accel_scan":
        backend = "adjoint"
        scan_backend = "accel_scan"

    if backend not in ("autograd", "adjoint"):
        raise ValueError(
            f"Unknown ParaGRU backend {backend!r}. Expected 'autograd' or 'adjoint'."
        )

    return DeerNewtonConfig(
        num_iters=num_iters,
        tol=tol,
        strict_tol=strict_tol,
        stopping_criterion="update",
        initial_guess=initial_guess,  # type: ignore[arg-type]
        quasi=True,
        scan_backend=scan_backend,
        accel_module=accel_module,
        jacobian_backend="explicit",
        backward_backend=backend,
    )


class ParaGRUCell(nn.Module):
    """Single-step diagonal ParaGRU cell, analogous to torch.nn.GRUCell.

    Args:
        input_size:
            Number of input features.
        hidden_size:
            Number of hidden-state features.
        bias:
            If False, the gate bias parameters are fixed to zero and are not
            trainable.

    Shapes:
        input: (input_size,) or (batch, input_size)
        hx: (hidden_size,) or (batch, hidden_size)
        output: same hidden layout as hx.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        recurrent_init_scale: float = 0.25,
        input_init_scale: float = 1.0,
        bias_init_value: float = 0.0,
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        factory_kwargs = {"device": device, "dtype": dtype}

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.bias_enabled = bool(bias)
        self.recurrent_init_scale = float(recurrent_init_scale)
        self.input_init_scale = float(input_init_scale)
        self.bias_init_value = float(bias_init_value)

        self.A = nn.Parameter(torch.empty(3, hidden_size, **factory_kwargs))
        self.B = nn.Parameter(torch.empty(
            3, input_size, hidden_size, **factory_kwargs))

        if self.bias_enabled:
            self.b = nn.Parameter(torch.empty(
                3, hidden_size, **factory_kwargs))
        else:
            self.register_buffer(
                "b",
                torch.zeros(3, hidden_size, **factory_kwargs),
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(
            self.A,
            a=-self.recurrent_init_scale,
            b=self.recurrent_init_scale,
        )

        for gate_idx in range(3):
            torch.nn.init.xavier_uniform_(self.B[gate_idx])
            if self.input_init_scale != 1.0:
                with torch.no_grad():
                    self.B[gate_idx].mul_(self.input_init_scale)

        if self.bias_enabled:
            torch.nn.init.constant_(self.b, self.bias_init_value)

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"bias={self.bias_enabled}"
        )

    def forward(
        self,
        input: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        unbatched = input.ndim == 1

        if input.ndim not in (1, 2):
            raise ValueError(
                "ParaGRUCell input must have shape (input_size,) or "
                f"(batch, input_size), got {tuple(input.shape)}."
            )

        if input.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected input.shape[-1] == {self.input_size}, got {input.shape[-1]}."
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
                    "ParaGRUCell hx must have shape (hidden_size,) or "
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
        """One ParaGRU recurrent update with argument order (state, input)."""
        return functional_paragru_recurrence_step(
            state=state,
            driver=driver,
            A=self.A,
            B=self.B,
            b=self.b,
        )

    def input_projection(self, driver: torch.Tensor) -> torch.Tensor:
        return functional_paragru_input_projection(driver, self.B, self.b)

    def compute_linearization_diag_from_previous(
        self,
        previous_states: torch.Tensor,
        drivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return functional_paragru_linearization_diag_from_previous(
            previous_states=previous_states,
            drivers=drivers,
            A=self.A,
            B=self.B,
            b=self.b,
        )

    def _compute_jacobians_diag_from_previous(
        self,
        previous_states: torch.Tensor,
        drivers: torch.Tensor,
    ) -> torch.Tensor:
        _, jacobian_diag = self.compute_linearization_diag_from_previous(
            previous_states=previous_states,
            drivers=drivers,
        )
        return jacobian_diag


class ParaGRU(BaseParaRNNCell):
    """Sequence-level diagonal ParaGRU, analogous to torch.nn.GRU.

    Public call:

        output, h_n = rnn(input, hx=None)

    Current implementation scope:
        * one recurrent layer,
        * one direction,
        * diagonal recurrent ParaGRU dynamics,
        * sequential or DEER forward modes,
        * optional custom adjoint backward for explicit quasi-DEER.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        *,
        mode: Literal["sequential", "deer"] = "deer",
        deer_config: DeerNewtonConfig | None = None,
        backend: ParaGRUBackend = "adjoint",
        scan_backend: Literal["torch", "accel_scan"] = "torch",
        num_iters: int = 4,
        tol: float | None = None,
        strict_tol: bool = False,
        accel_module: str = "warp",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        recurrent_init_scale: float = 0.25,
        input_init_scale: float = 1.0,
        bias_init_value: float = 0.0,
    ):
        if num_layers != 1:
            raise NotImplementedError(
                "ParaGRU currently supports num_layers=1 only.")
        if bidirectional:
            raise NotImplementedError(
                "ParaGRU currently supports bidirectional=False only.")
        if dropout != 0.0:
            raise NotImplementedError(
                "ParaGRU currently supports dropout=0.0 only.")

        if deer_config is None:
            deer_config = make_paragru_deer_config(
                backend=backend,
                num_iters=num_iters,
                tol=tol,
                strict_tol=strict_tol,
                scan_backend=scan_backend,
                accel_module=accel_module,
            )

        config = ParaGRUConfig(
            input_dim=input_size,
            state_dim=hidden_size,
            output_dim=hidden_size,
            mode=mode,
            batch_first=batch_first,
            device=torch.device(device) if device is not None else None,
            dtype=dtype,
            deer=deer_config,
            recurrent_init_scale=recurrent_init_scale,
            input_init_scale=input_init_scale,
            bias_init_value=bias_init_value,
            bias=bias,
        )

        super().__init__(config)

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bias = bool(bias)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)

        self.cell = ParaGRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            device=device,
            dtype=dtype,
            recurrent_init_scale=recurrent_init_scale,
            input_init_scale=input_init_scale,
            bias_init_value=bias_init_value,
        )

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, bias={self.bias}, "
            f"batch_first={self.batch_first}, mode={self.mode}"
        )

    def recurrence_step(
        self,
        state: torch.Tensor,
        driver: torch.Tensor,
    ) -> torch.Tensor:
        return self.cell.recurrence_step(state, driver)

    def reset_parameters(self) -> None:
        self.cell.reset_parameters()

    @property
    def A(self) -> torch.nn.Parameter:
        return self.cell.A

    @property
    def B(self) -> torch.nn.Parameter:
        return self.cell.B

    @property
    def b(self) -> torch.Tensor:
        return self.cell.b

    def forward(
        self,
        input: torch.Tensor,
        hx: torch.Tensor | None = None,
        *,
        mode: Literal["sequential", "deer"] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the ParaGRU over a full sequence.

        Args:
            input:
                (L, input_size) for unbatched input,
                (L, N, input_size) when batch_first=False, or
                (N, L, input_size) when batch_first=True.
            hx:
                Optional initial state with shape (1, H) for unbatched input, or
                (1, N, H) for batched input. (H,) and (N, H) are also accepted
                for convenience.

        Returns:
            (output, h_n), following the PyTorch GRU convention.
        """
        initial_state, unbatched_input = self._hx_to_initial_state(input, hx)
        selected_mode = self.mode if mode is None else mode

        if selected_mode == "sequential":
            output = self.forward_sequential(
                input, initial_state=initial_state)
        elif selected_mode == "deer":
            output = self.forward_deer(input, initial_state=initial_state)
        else:
            raise ValueError(
                f"Unknown mode {selected_mode!r}. Expected 'sequential' or 'deer'."
            )

        h_n = self._make_h_n(output, unbatched_input=unbatched_input)
        return output, h_n

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
                "ParaGRU input must have shape (L, input_size), "
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
                    f"Expected hx shape {expected}, got {tuple(hx.shape)}.")
            return hx, False

        if hx.ndim == 3:
            expected = (1, batch_size, self.hidden_size)
            if tuple(hx.shape) != expected:
                raise ValueError(
                    f"Expected hx shape {expected}, got {tuple(hx.shape)}.")
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

    def forward_deer(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        deer_config: DeerNewtonConfig | None = None,
    ) -> torch.Tensor:
        cfg = self.config.deer if deer_config is None else deer_config

        if getattr(cfg, "backward_backend", "autograd") != "adjoint":
            return super().forward_deer(
                x=x,
                initial_state=initial_state,
                deer_config=cfg,
            )

        self._validate_adjoint_deer_config(cfg)

        x_batched, had_batch_dim = self._normalize_input(x)
        initial_state_batched = self._normalize_initial_state(
            x_batched=x_batched,
            initial_state=initial_state,
        )
        accel_scan_fn = self._load_accel_scan_if_needed(cfg)

        states = paragru_adjoint_deer_forward(
            drivers=x_batched,
            initial_state=initial_state_batched,
            A=self.cell.A,
            B=self.cell.B,
            b=self.cell.b,
            cfg=cfg,
            accel_scan_fn=accel_scan_fn,
        )

        with torch.no_grad():
            final_merit = merit_fxn_batched(
                f=self.recurrence_step,
                initial_state=initial_state_batched,
                states=states.detach(),
                drivers=x_batched,
            )

        self.last_deer_infos = [
            {
                "num_iters": cfg.num_iters,
                "initial_merit": None,
                "final_merit": final_merit.detach(),
                "last_update_error": None,
                "tol": cfg.tol,
                "effective_tol": None,
                "strict_tol": cfg.strict_tol,
                "stopping_criterion": cfg.stopping_criterion,
                "scan_backend": cfg.scan_backend,
                "quasi": True,
                "batched": True,
                "batch_size": x_batched.shape[0],
                "jacobian_backend": "custom",
                "linearization_backend": "custom",
                "backward_backend": "adjoint",
            }
        ]

        outputs = self.post_process(states)

        return self._restore_output_layout(
            outputs,
            had_batch_dim=had_batch_dim,
        )

    @staticmethod
    def _validate_adjoint_deer_config(cfg: DeerNewtonConfig) -> None:
        if not cfg.quasi:
            raise ValueError(
                "backward_backend='adjoint' currently requires quasi=True.")
        if cfg.jacobian_backend != "explicit":
            raise ValueError(
                "backward_backend='adjoint' currently requires "
                "jacobian_backend='explicit'."
            )
        if cfg.return_trace:
            raise ValueError(
                "backward_backend='adjoint' does not support return_trace=True.")

    def assemble_initial_guess(
        self,
        drivers: torch.Tensor,
        initial_state: torch.Tensor,
        guess_type: str = "f0",
    ) -> torch.Tensor:
        if guess_type == "zero":
            return torch.zeros(
                drivers.shape[0],
                self.hidden_size,
                device=drivers.device,
                dtype=drivers.dtype,
            )

        if guess_type == "f0":
            zero_states = torch.zeros(
                drivers.shape[0],
                self.hidden_size,
                device=drivers.device,
                dtype=drivers.dtype,
            )
            return self.recurrence_step(zero_states, drivers)

        raise ValueError(f"Unknown initial guess type: {guess_type!r}.")

    def compute_jacobians_diag(
        self,
        states: torch.Tensor,
        drivers: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_batched, _ = self._normalize_input(drivers)
        state_batched, _ = self._normalize_states(states)
        initial_state_batched = self._normalize_initial_state(
            x_batched=x_batched,
            initial_state=initial_state,
        )
        previous_states = self.roll_state(
            states=state_batched,
            initial_state=initial_state_batched,
        )
        return self._compute_jacobians_diag_from_previous(
            previous_states=previous_states,
            drivers=x_batched,
        )

    def compute_jacobians_bwd_diag(
        self,
        states: torch.Tensor,
        drivers: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        jac = self.compute_jacobians_diag(
            states=states,
            drivers=drivers,
            initial_state=initial_state,
        )
        jac_bwd = torch.roll(torch.flip(jac, dims=[1]), shifts=1, dims=1)
        jac_bwd[:, 0, :] = 0.0
        return jac_bwd

    def compute_linearization_diag_from_previous(
        self,
        previous_states: torch.Tensor,
        drivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cell.compute_linearization_diag_from_previous(
            previous_states=previous_states,
            drivers=drivers,
        )

    def _compute_jacobians_diag_from_previous(
        self,
        previous_states: torch.Tensor,
        drivers: torch.Tensor,
    ) -> torch.Tensor:
        return self.cell._compute_jacobians_diag_from_previous(
            previous_states=previous_states,
            drivers=drivers,
        )
