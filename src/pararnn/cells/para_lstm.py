from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from src.pararnn.base_cell import BaseParaRNNCell
from src.pararnn.config import DeerNewtonConfig, ParaRNNConfig
from src.utils.BlockScan import block2_mat_scan


ParaLSTMBackend = Literal["autograd", "block_deer_autograd_torch"]


@dataclass
class ParaLSTMConfig(ParaRNNConfig):
    """Internal config for ParaLSTM.

    The internal state is concat(c, h), so state_dim = 2 * hidden_size.
    The user-facing output is h, so output_dim = hidden_size.
    """

    hidden_size: int = 0
    recurrent_init_scale: float = 0.25
    input_init_scale: float = 1.0
    bias_init_value: float = 0.0
    forget_bias_init_value: float = 1.0
    bias: bool = True


def make_paralstm_deer_config(
    backend: ParaLSTMBackend = "autograd",
    *,
    num_iters: int = 4,
    tol: float | None = None,
    strict_tol: bool = False,
    initial_guess: str = "f0",
    scan_backend: Literal["torch"] = "torch",
) -> DeerNewtonConfig:
    """Construct a block-DEER config for ParaLSTM.

    LSTM has state (c_t, h_t). Therefore each hidden coordinate has a 2D state,
    and the structured Jacobian is block diagonal with 2x2 blocks.
    """
    if backend == "block_deer_autograd_torch":
        backend = "autograd"

    if backend != "autograd":
        raise ValueError(
            f"Unknown ParaLSTM backend {backend!r}. Expected 'autograd'."
        )

    if scan_backend != "torch":
        raise ValueError(
            "ParaLSTM block-DEER currently supports scan_backend='torch' only."
        )

    return DeerNewtonConfig(
        num_iters=num_iters,
        tol=tol,
        strict_tol=strict_tol,
        stopping_criterion="update",
        initial_guess=initial_guess,  # type: ignore[arg-type]
        quasi=True,
        scan_backend="torch",
        accel_module="warp",
        jacobian_backend="explicit",
        backward_backend="autograd",
    )


def functional_paralstm_input_projection(
    driver: torch.Tensor,
    B: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("...i,gij->...gj", driver, B) + b


def _split_flat_state(
    state: torch.Tensor,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    c = state[..., :hidden_size]
    h = state[..., hidden_size:]
    return c, h


def _pack_flat_state(c: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return torch.cat([c, h], dim=-1)


def _flat_to_blocks(state: torch.Tensor, hidden_size: int) -> torch.Tensor:
    c, h = _split_flat_state(state, hidden_size)
    return torch.stack([c, h], dim=-1)


def _blocks_to_flat(block_state: torch.Tensor) -> torch.Tensor:
    if block_state.shape[-1] != 2:
        raise ValueError(
            "block_state must end with dimension 2 containing (c, h), got "
            f"{tuple(block_state.shape)}."
        )

    c = block_state[..., 0]
    h = block_state[..., 1]
    return _pack_flat_state(c, h)


def functional_paralstm_recurrence_step(
    state: torch.Tensor,
    driver: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Functional diagonal-recurrent LSTM step.

    Internal state:

        state = concat(c, h)

    Recurrent parameters A are diagonal and act on h_prev.
    """
    hidden_size = A.shape[1]
    c_prev, h_prev = _split_flat_state(state, hidden_size)

    Bx_plus_b = functional_paralstm_input_projection(driver, B, b)

    i_pre = A[0] * h_prev + Bx_plus_b[..., 0, :]
    f_pre = A[1] * h_prev + Bx_plus_b[..., 1, :]
    g_pre = A[2] * h_prev + Bx_plus_b[..., 2, :]
    o_pre = A[3] * h_prev + Bx_plus_b[..., 3, :]

    i = torch.sigmoid(i_pre)
    f = torch.sigmoid(f_pre)
    g = torch.tanh(g_pre)
    o = torch.sigmoid(o_pre)

    c_next = f * c_prev + i * g
    h_next = o * torch.tanh(c_next)

    return _pack_flat_state(c_next, h_next)


def functional_paralstm_linearization_blocks_from_previous(
    previous_states: torch.Tensor,
    drivers: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predicted block states and exact 2x2 block Jacobians.

    Args:
        previous_states: (B, T, H, 2), final dimension is (c_prev, h_prev)
        drivers: (B, T, input_size)

    Returns:
        predicted_states: (B, T, H, 2)
        jacobian_blocks: (B, T, H, 2, 2)
    """
    if previous_states.ndim != 4 or previous_states.shape[-1] != 2:
        raise ValueError(
            "previous_states must have shape (B, T, H, 2), got "
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

    c_prev = previous_states[..., 0]
    h_prev = previous_states[..., 1]

    Bx_plus_b = functional_paralstm_input_projection(drivers, B, b)

    i_pre = A[0] * h_prev + Bx_plus_b[..., 0, :]
    f_pre = A[1] * h_prev + Bx_plus_b[..., 1, :]
    g_pre = A[2] * h_prev + Bx_plus_b[..., 2, :]
    o_pre = A[3] * h_prev + Bx_plus_b[..., 3, :]

    i = torch.sigmoid(i_pre)
    f = torch.sigmoid(f_pre)
    g = torch.tanh(g_pre)
    o = torch.sigmoid(o_pre)

    c_next = f * c_prev + i * g
    tanh_c = torch.tanh(c_next)
    h_next = o * tanh_c

    di_dh = A[0] * i * (1.0 - i)
    df_dh = A[1] * f * (1.0 - f)
    dg_dh = A[2] * (1.0 - g * g)
    do_dh = A[3] * o * (1.0 - o)

    dc_dc = f
    dc_dh = c_prev * df_dh + g * di_dh + i * dg_dh

    dh_dc = o * (1.0 - tanh_c * tanh_c) * dc_dc
    dh_dh = do_dh * tanh_c + o * (1.0 - tanh_c * tanh_c) * dc_dh

    predicted_states = torch.stack([c_next, h_next], dim=-1)

    row0 = torch.stack([dc_dc, dc_dh], dim=-1)
    row1 = torch.stack([dh_dc, dh_dh], dim=-1)
    jacobian_blocks = torch.stack([row0, row1], dim=-2)

    return predicted_states, jacobian_blocks


def _dtype_default_tol(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return 1e-4
    if dtype == torch.float64:
        return 1e-7
    return 1e-7


def _effective_tol(
    dtype: torch.dtype,
    tol: float | None,
    strict_tol: bool,
) -> float:
    if tol is None:
        return _dtype_default_tol(dtype)

    tol = float(tol)

    if strict_tol:
        return tol

    return max(tol, _dtype_default_tol(dtype))


class ParaLSTMCell(nn.Module):
    """Single-step diagonal ParaLSTM cell, analogous to torch.nn.LSTMCell."""

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
        forget_bias_init_value: float = 1.0,
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
        self.forget_bias_init_value = float(forget_bias_init_value)

        self.A = nn.Parameter(torch.empty(4, hidden_size, **factory_kwargs))
        self.B = nn.Parameter(
            torch.empty(4, input_size, hidden_size, **factory_kwargs)
        )

        if self.bias_enabled:
            self.b = nn.Parameter(torch.empty(4, hidden_size, **factory_kwargs))
        else:
            self.register_buffer(
                "b",
                torch.zeros(4, hidden_size, **factory_kwargs),
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(
            self.A,
            a=-self.recurrent_init_scale,
            b=self.recurrent_init_scale,
        )

        for gate_idx in range(4):
            torch.nn.init.xavier_uniform_(self.B[gate_idx])
            if self.input_init_scale != 1.0:
                with torch.no_grad():
                    self.B[gate_idx].mul_(self.input_init_scale)

        if self.bias_enabled:
            torch.nn.init.constant_(self.b, self.bias_init_value)
            with torch.no_grad():
                self.b[1].fill_(self.forget_bias_init_value)

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"bias={self.bias_enabled}"
        )

    def forward(
        self,
        input: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unbatched = input.ndim == 1

        if input.ndim not in (1, 2):
            raise ValueError(
                "ParaLSTMCell input must have shape (input_size,) or "
                f"(batch, input_size), got {tuple(input.shape)}."
            )

        if input.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected input.shape[-1] == {self.input_size}, "
                f"got {input.shape[-1]}."
            )

        input_batched = input.unsqueeze(0) if unbatched else input
        batch_size = input_batched.shape[0]

        if hx is None:
            h_batched = torch.zeros(
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            )
            c_batched = torch.zeros_like(h_batched)
        else:
            if not isinstance(hx, tuple) or len(hx) != 2:
                raise ValueError("ParaLSTMCell hx must be None or a tuple (h, c).")
            h_batched = self._normalize_cell_state(hx[0], input_batched, "h")
            c_batched = self._normalize_cell_state(hx[1], input_batched, "c")

        state = _pack_flat_state(c_batched, h_batched)
        next_state = self.recurrence_step(state, input_batched)
        c_next, h_next = _split_flat_state(next_state, self.hidden_size)

        if unbatched:
            return h_next.squeeze(0), c_next.squeeze(0)

        return h_next, c_next

    def _normalize_cell_state(
        self,
        state: torch.Tensor,
        input_batched: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        elif state.ndim != 2:
            raise ValueError(
                f"ParaLSTMCell {name} must have shape (hidden_size,) or "
                f"(batch, hidden_size), got {tuple(state.shape)}."
            )

        expected = (input_batched.shape[0], self.hidden_size)
        if tuple(state.shape) != expected:
            raise ValueError(
                f"Expected {name} shape {expected}, got {tuple(state.shape)}."
            )

        return state.to(device=input_batched.device, dtype=input_batched.dtype)

    def recurrence_step(
        self,
        state: torch.Tensor,
        driver: torch.Tensor,
    ) -> torch.Tensor:
        return functional_paralstm_recurrence_step(
            state=state,
            driver=driver,
            A=self.A,
            B=self.B,
            b=self.b,
        )

    def input_projection(self, driver: torch.Tensor) -> torch.Tensor:
        return functional_paralstm_input_projection(driver, self.B, self.b)

    def compute_linearization_blocks_from_previous(
        self,
        previous_states: torch.Tensor,
        drivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return functional_paralstm_linearization_blocks_from_previous(
            previous_states=previous_states,
            drivers=drivers,
            A=self.A,
            B=self.B,
            b=self.b,
        )


class ParaLSTM(BaseParaRNNCell):
    """Sequence-level diagonal ParaLSTM, analogous to torch.nn.LSTM."""

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
        mode: Literal["sequential", "deer"] = "sequential",
        deer_config: DeerNewtonConfig | None = None,
        backend: ParaLSTMBackend = "autograd",
        scan_backend: Literal["torch"] = "torch",
        num_iters: int = 4,
        tol: float | None = None,
        strict_tol: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        recurrent_init_scale: float = 0.25,
        input_init_scale: float = 1.0,
        bias_init_value: float = 0.0,
        forget_bias_init_value: float = 1.0,
    ):
        if num_layers != 1:
            raise NotImplementedError(
                "ParaLSTM currently supports num_layers=1 only."
            )
        if bidirectional:
            raise NotImplementedError(
                "ParaLSTM currently supports bidirectional=False only."
            )
        if dropout != 0.0:
            raise NotImplementedError("ParaLSTM currently supports dropout=0.0 only.")

        if deer_config is None:
            deer_config = make_paralstm_deer_config(
                backend=backend,
                num_iters=num_iters,
                tol=tol,
                strict_tol=strict_tol,
                scan_backend=scan_backend,
            )

        config = ParaLSTMConfig(
            input_dim=input_size,
            state_dim=2 * hidden_size,
            output_dim=hidden_size,
            mode=mode,
            batch_first=batch_first,
            device=torch.device(device) if device is not None else None,
            dtype=dtype,
            deer=deer_config,
            hidden_size=hidden_size,
            recurrent_init_scale=recurrent_init_scale,
            input_init_scale=input_init_scale,
            bias_init_value=bias_init_value,
            forget_bias_init_value=forget_bias_init_value,
            bias=bias,
        )

        super().__init__(config)

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bias = bool(bias)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)

        self.cell = ParaLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            device=device,
            dtype=dtype,
            recurrent_init_scale=recurrent_init_scale,
            input_init_scale=input_init_scale,
            bias_init_value=bias_init_value,
            forget_bias_init_value=forget_bias_init_value,
        )

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, bias={self.bias}, "
            f"batch_first={self.batch_first}, mode={self.mode}"
        )

    @property
    def A(self) -> torch.nn.Parameter:
        return self.cell.A

    @property
    def B(self) -> torch.nn.Parameter:
        return self.cell.B

    @property
    def b(self) -> torch.Tensor:
        return self.cell.b

    def reset_parameters(self) -> None:
        self.cell.reset_parameters()

    def recurrence_step(
        self,
        state: torch.Tensor,
        driver: torch.Tensor,
    ) -> torch.Tensor:
        return self.cell.recurrence_step(state, driver)

    def post_process(self, states: torch.Tensor) -> torch.Tensor:
        _, h = _split_flat_state(states, self.hidden_size)
        return h

    def forward(
        self,
        input: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        mode: Literal["sequential", "deer"] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x_batched, had_batch_dim = self._normalize_input(input)
        unbatched_input = not had_batch_dim

        initial_state = self._normalize_lstm_hx(
            x_batched=x_batched,
            hx=hx,
            unbatched_input=unbatched_input,
        )

        selected_mode = self.mode if mode is None else mode

        if selected_mode == "sequential":
            states = self.batched_sequential_rollout(
                initial_state=initial_state,
                drivers=x_batched,
            )
        elif selected_mode == "deer":
            states = self.forward_deer_states(
                x_batched=x_batched,
                initial_state=initial_state,
                deer_config=self.config.deer,
            )
        else:
            raise ValueError(
                f"Unknown mode {selected_mode!r}. Expected 'sequential' or 'deer'."
            )

        outputs_batched = self.post_process(states)
        output = self._restore_output_layout(
            outputs_batched,
            had_batch_dim=had_batch_dim,
        )
        h_n, c_n = self._make_h_c_n(states, unbatched_input=unbatched_input)

        return output, (h_n, c_n)

    def _normalize_lstm_hx(
        self,
        x_batched: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        unbatched_input: bool,
    ) -> torch.Tensor:
        batch_size = x_batched.shape[0]
        device = x_batched.device
        dtype = x_batched.dtype

        if hx is None:
            h0 = torch.zeros(
                batch_size,
                self.hidden_size,
                device=device,
                dtype=dtype,
            )
            c0 = torch.zeros_like(h0)
            return _pack_flat_state(c0, h0)

        if not isinstance(hx, tuple) or len(hx) != 2:
            raise ValueError("ParaLSTM hx must be None or a tuple (h_0, c_0).")

        h0 = self._normalize_one_lstm_hx_tensor(
            hx[0],
            batch_size=batch_size,
            unbatched_input=unbatched_input,
            name="h_0",
            device=device,
            dtype=dtype,
        )
        c0 = self._normalize_one_lstm_hx_tensor(
            hx[1],
            batch_size=batch_size,
            unbatched_input=unbatched_input,
            name="c_0",
            device=device,
            dtype=dtype,
        )

        return _pack_flat_state(c0, h0)

    def _normalize_one_lstm_hx_tensor(
        self,
        tensor: torch.Tensor,
        *,
        batch_size: int,
        unbatched_input: bool,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = tensor.to(device=device, dtype=dtype)

        if unbatched_input:
            if tensor.ndim == 1:
                expected = (self.hidden_size,)
                if tuple(tensor.shape) != expected:
                    raise ValueError(
                        f"Expected {name} shape {expected}, "
                        f"got {tuple(tensor.shape)}."
                    )
                return tensor.unsqueeze(0)

            if tensor.ndim == 2:
                expected = (1, self.hidden_size)
                if tuple(tensor.shape) != expected:
                    raise ValueError(
                        f"Expected {name} shape {expected}, "
                        f"got {tuple(tensor.shape)}."
                    )
                return tensor

            raise ValueError(
                f"For unbatched input, {name} must have shape (hidden_size,) "
                "or (1, hidden_size)."
            )

        if tensor.ndim == 2:
            expected = (batch_size, self.hidden_size)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"Expected {name} shape {expected}, "
                    f"got {tuple(tensor.shape)}."
                )
            return tensor

        if tensor.ndim == 3:
            expected = (1, batch_size, self.hidden_size)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"Expected {name} shape {expected}, "
                    f"got {tuple(tensor.shape)}."
                )
            return tensor[0]

        raise ValueError(
            f"For batched input, {name} must have shape "
            "(batch, hidden_size) or (1, batch, hidden_size)."
        )

    def _make_h_c_n(
        self,
        states: torch.Tensor,
        *,
        unbatched_input: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states.shape[1] == 0:
            raise ValueError("Cannot compute h_n and c_n for an empty sequence.")

        final_state = states[:, -1, :]
        c_final, h_final = _split_flat_state(final_state, self.hidden_size)

        if unbatched_input:
            return (
                h_final.squeeze(0).unsqueeze(0),
                c_final.squeeze(0).unsqueeze(0),
            )

        return h_final.unsqueeze(0), c_final.unsqueeze(0)

    def _assemble_initial_guess_blocks(
        self,
        drivers: torch.Tensor,
        guess_type: str,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = drivers.shape
        zeros = torch.zeros(
            batch_size,
            seq_len,
            self.hidden_size,
            2,
            device=drivers.device,
            dtype=drivers.dtype,
        )

        if guess_type == "zero":
            return zeros

        if guess_type == "f0":
            predicted, _ = self.cell.compute_linearization_blocks_from_previous(
                previous_states=zeros,
                drivers=drivers,
            )
            return predicted

        raise ValueError(f"Unknown initial guess type: {guess_type!r}.")

    def forward_deer_states(
        self,
        x_batched: torch.Tensor,
        initial_state: torch.Tensor,
        deer_config: DeerNewtonConfig,
    ) -> torch.Tensor:
        cfg = deer_config
        self._validate_block_deer_config(cfg)

        initial_blocks = _flat_to_blocks(initial_state, self.hidden_size)

        states = self._assemble_initial_guess_blocks(
            drivers=x_batched,
            guess_type=cfg.initial_guess,
        )

        effective_tol = _effective_tol(
            dtype=states.dtype,
            tol=cfg.tol,
            strict_tol=cfg.strict_tol,
        )

        initial_merit = self._block_deer_merit(
            initial_blocks=initial_blocks,
            states=states,
            drivers=x_batched,
        )

        last_update_error = torch.tensor(
            float("inf"),
            device=states.device,
            dtype=states.dtype,
        )
        num_steps_done = 0

        for iter_idx in range(cfg.num_iters):
            old_states = states

            previous_states = torch.cat(
                [initial_blocks[:, None, :, :], old_states[:, :-1, :, :]],
                dim=1,
            )

            predicted, jacobian_blocks = (
                self.cell.compute_linearization_blocks_from_previous(
                    previous_states=previous_states,
                    drivers=x_batched,
                )
            )

            b_terms = predicted - (
                jacobian_blocks @ previous_states.unsqueeze(-1)
            ).squeeze(-1)

            A0 = torch.zeros_like(jacobian_blocks[:, 0, :, :, :])
            b0 = predicted[:, 0, :, :]

            if predicted.shape[1] > 1:
                A_scan = torch.cat(
                    [A0[:, None, :, :, :], jacobian_blocks[:, 1:, :, :, :]],
                    dim=1,
                )
                b_scan = torch.cat(
                    [b0[:, None, :, :], b_terms[:, 1:, :, :]],
                    dim=1,
                )
            else:
                A_scan = A0[:, None, :, :, :]
                b_scan = b0[:, None, :, :]

            _, states = block2_mat_scan(A_scan, b_scan, dim=1)

            if cfg.clip_value is not None:
                states = torch.clamp(states, -cfg.clip_value, cfg.clip_value)
                states = torch.nan_to_num(states)

            last_update_error = torch.max(torch.abs(states - old_states))
            num_steps_done = iter_idx + 1

            if (
                cfg.stopping_criterion == "update"
                and last_update_error.item() <= effective_tol
            ):
                break

            if cfg.stopping_criterion == "merit":
                current_merit = self._block_deer_merit(
                    initial_blocks=initial_blocks,
                    states=states,
                    drivers=x_batched,
                )
                if current_merit.item() <= effective_tol:
                    break

        final_merit = self._block_deer_merit(
            initial_blocks=initial_blocks,
            states=states,
            drivers=x_batched,
        )

        self.last_deer_infos = [
            {
                "num_iters": num_steps_done,
                "initial_merit": initial_merit.detach(),
                "final_merit": final_merit.detach(),
                "last_update_error": last_update_error.detach(),
                "tol": cfg.tol,
                "effective_tol": effective_tol,
                "strict_tol": cfg.strict_tol,
                "stopping_criterion": cfg.stopping_criterion,
                "scan_backend": "torch_block2_associative_scan",
                "quasi": True,
                "batched": True,
                "batch_size": x_batched.shape[0],
                "jacobian_backend": "explicit_block2",
                "linearization_backend": "custom_block2",
                "backward_backend": "autograd",
            }
        ]

        return _blocks_to_flat(states)

    def _block_deer_merit(
        self,
        initial_blocks: torch.Tensor,
        states: torch.Tensor,
        drivers: torch.Tensor,
    ) -> torch.Tensor:
        previous_states = torch.cat(
            [initial_blocks[:, None, :, :], states[:, :-1, :, :]],
            dim=1,
        )
        predicted, _ = self.cell.compute_linearization_blocks_from_previous(
            previous_states=previous_states,
            drivers=drivers,
        )
        residual = states - predicted
        return 0.5 * torch.sum(residual * residual)

    @staticmethod
    def _validate_block_deer_config(cfg: DeerNewtonConfig) -> None:
        if not cfg.quasi:
            raise ValueError("ParaLSTM block-DEER currently requires quasi=True.")
        if cfg.scan_backend != "torch":
            raise ValueError(
                "ParaLSTM block-DEER currently supports scan_backend='torch' only."
            )
        if cfg.jacobian_backend != "explicit":
            raise ValueError(
                "ParaLSTM block-DEER requires jacobian_backend='explicit'."
            )
        if cfg.backward_backend != "autograd":
            raise ValueError(
                "ParaLSTM block-DEER currently supports "
                "backward_backend='autograd' only."
            )
        if cfg.return_trace:
            raise ValueError(
                "ParaLSTM block-DEER does not support return_trace=True yet."
            )
        if cfg.stopping_criterion not in ("update", "merit"):
            raise ValueError("stopping_criterion must be 'update' or 'merit'.")
