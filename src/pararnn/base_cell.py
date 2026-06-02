from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.func import jacrev, vmap

from src.algos.DEER import deer_alg_batched
from src.pararnn.config import DeerNewtonConfig, ParaRNNConfig


class BaseParaRNNCell(torch.nn.Module, abc.ABC):
    def __init__(self, config: ParaRNNConfig):
        super().__init__()

        self.config = config
        self.input_dim = int(config.input_dim)
        self.state_dim = int(config.state_dim)
        self.output_dim = int(config.output_dim)
        self.mode = config.mode
        self.batch_first = bool(config.batch_first)
        self.last_deer_infos: List[Dict[str, Any]] = []

    @abc.abstractmethod
    def recurrence_step(
        self,
        state: torch.Tensor,
        driver: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def post_process(self, states: torch.Tensor) -> torch.Tensor:
        return states

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        selected_mode = self.mode if mode is None else mode

        if selected_mode == "sequential":
            return self.forward_sequential(
                x=x,
                initial_state=initial_state,
            )

        if selected_mode == "deer":
            return self.forward_deer(
                x=x,
                initial_state=initial_state,
            )

        raise ValueError(
            f"Unknown mode {selected_mode!r}. Expected 'sequential' or 'deer'."
        )

    def forward_sequential(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_batched, had_batch_dim = self._normalize_input(x)
        initial_state_batched = self._normalize_initial_state(
            x_batched=x_batched,
            initial_state=initial_state,
        )

        states = self.batched_sequential_rollout(
            initial_state=initial_state_batched,
            drivers=x_batched,
        )

        outputs = self.post_process(states)

        return self._restore_output_layout(
            outputs,
            had_batch_dim=had_batch_dim,
        )

    def forward_deer(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        deer_config: Optional[DeerNewtonConfig] = None,
    ) -> torch.Tensor:
        cfg = self.config.deer if deer_config is None else deer_config

        x_batched, had_batch_dim = self._normalize_input(x)
        initial_state_batched = self._normalize_initial_state(
            x_batched=x_batched,
            initial_state=initial_state,
        )

        accel_scan_fn = self._load_accel_scan_if_needed(cfg)
        jacobian_fn = self._make_deer_jacobian_fn(cfg)
        linearization_fn = self._make_deer_linearization_fn(cfg)

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
            quasi=cfg.quasi,
            damping=cfg.damping,
            clip_value=cfg.clip_value,
            return_trace=cfg.return_trace,
            scan_backend=cfg.scan_backend,
            accel_scan_fn=accel_scan_fn,
            strict_tol=cfg.strict_tol,
            stopping_criterion=cfg.stopping_criterion,
            jacobian_fn=jacobian_fn,
            linearization_fn=linearization_fn,
        )

        self.last_deer_infos = [info]

        outputs = self.post_process(states)

        return self._restore_output_layout(
            outputs,
            had_batch_dim=had_batch_dim,
        )

    def forward_states_sequential(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_batched, had_batch_dim = self._normalize_input(x)
        initial_state_batched = self._normalize_initial_state(
            x_batched=x_batched,
            initial_state=initial_state,
        )

        states = self.batched_sequential_rollout(
            initial_state=initial_state_batched,
            drivers=x_batched,
        )

        return self._restore_output_layout(
            states,
            had_batch_dim=had_batch_dim,
        )

    def batched_sequential_rollout(
        self,
        initial_state: torch.Tensor,
        drivers: torch.Tensor,
    ) -> torch.Tensor:
        if drivers.ndim != 3:
            raise ValueError(
                "drivers must have shape (B, T, input_dim), "
                f"got {tuple(drivers.shape)}."
            )

        if initial_state.ndim != 2:
            raise ValueError(
                "initial_state must have shape (B, state_dim), "
                f"got {tuple(initial_state.shape)}."
            )

        batch_size, seq_len, input_dim = drivers.shape

        if input_dim != self.input_dim:
            raise ValueError(
                f"Expected drivers.shape[-1] == {self.input_dim}, got {input_dim}."
            )

        if initial_state.shape != (batch_size, self.state_dim):
            raise ValueError(
                "initial_state must have shape (B, state_dim), got "
                f"{tuple(initial_state.shape)} for B={batch_size}, "
                f"state_dim={self.state_dim}."
            )

        state = initial_state
        states = []

        for time_idx in range(seq_len):
            state = self.recurrence_step(
                state,
                drivers[:, time_idx, :],
            )

            if state.shape != (batch_size, self.state_dim):
                raise RuntimeError(
                    "recurrence_step must return shape (B, state_dim), got "
                    f"{tuple(state.shape)}."
                )

            states.append(state)

        if seq_len == 0:
            return torch.empty(
                batch_size,
                0,
                self.state_dim,
                device=drivers.device,
                dtype=drivers.dtype,
            )

        return torch.stack(states, dim=1)

    def assemble_initial_guess(
        self,
        drivers: torch.Tensor,
        initial_state: torch.Tensor,
        guess_type: str = "f0",
    ) -> torch.Tensor:
        if guess_type == "zero":
            return torch.zeros(
                drivers.shape[0],
                self.state_dim,
                device=drivers.device,
                dtype=drivers.dtype,
            )

        if guess_type == "f0":
            zero_state = torch.zeros_like(initial_state)
            guesses = [
                self.recurrence_step(zero_state, drivers[t])
                for t in range(drivers.shape[0])
            ]
            return torch.stack(guesses, dim=0)

        raise ValueError(f"Unknown initial guess type: {guess_type!r}.")

    def assemble_initial_guess_batched(
        self,
        drivers: torch.Tensor,
        initial_state: torch.Tensor,
        guess_type: str = "f0",
    ) -> torch.Tensor:
        if drivers.ndim != 3:
            raise ValueError(
                "drivers must have shape (B, T, input_dim), "
                f"got {tuple(drivers.shape)}."
            )

        if initial_state.ndim != 2:
            raise ValueError(
                "initial_state must have shape (B, state_dim), "
                f"got {tuple(initial_state.shape)}."
            )

        batch_size, seq_len, input_dim = drivers.shape

        if input_dim != self.input_dim:
            raise ValueError(
                f"Expected drivers.shape[-1] == {self.input_dim}, got {input_dim}."
            )

        if initial_state.shape != (batch_size, self.state_dim):
            raise ValueError(
                "initial_state must have shape (B, state_dim), got "
                f"{tuple(initial_state.shape)} for B={batch_size}, "
                f"state_dim={self.state_dim}."
            )

        if guess_type == "zero":
            return torch.zeros(
                batch_size,
                seq_len,
                self.state_dim,
                device=drivers.device,
                dtype=drivers.dtype,
            )

        if guess_type == "f0":
            zero_states = torch.zeros(
                batch_size,
                seq_len,
                self.state_dim,
                device=drivers.device,
                dtype=drivers.dtype,
            )
            return self.batched_recurrence_step(
                previous_states=zero_states,
                drivers=drivers,
            )

        raise ValueError(f"Unknown initial guess type: {guess_type!r}.")

    def initial_guess(
        self,
        drivers: torch.Tensor,
        initial_state: torch.Tensor,
        guess_type: str = "f0",
    ) -> torch.Tensor:
        return self.assemble_initial_guess(
            drivers=drivers,
            initial_state=initial_state,
            guess_type=guess_type,
        )

    def roll_state(
        self,
        states: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError(
                f"Expected states shape (B, T, D), got {tuple(states.shape)}."
            )

        if initial_state.ndim != 2:
            raise ValueError(
                f"Expected initial_state shape (B, D), got {tuple(initial_state.shape)}."
            )

        return torch.cat(
            [initial_state[:, None, :], states[:, :-1, :]],
            dim=1,
        )

    def batched_recurrence_step(
        self,
        previous_states: torch.Tensor,
        drivers: torch.Tensor,
    ) -> torch.Tensor:
        if previous_states.shape[:-1] != drivers.shape[:-1]:
            raise ValueError(
                "previous_states and drivers must share batch/time dimensions, "
                f"got {tuple(previous_states.shape)} and {tuple(drivers.shape)}."
            )

        flat_prev = previous_states.reshape(-1, self.state_dim)
        flat_drivers = drivers.reshape(-1, self.input_dim)

        predicted_flat = vmap(self.recurrence_step)(flat_prev, flat_drivers)

        return predicted_flat.reshape(
            *previous_states.shape[:-1],
            self.state_dim,
        )

    def compute_negative_residuals(
        self,
        states: torch.Tensor,
        drivers: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
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

        predicted_states = self.batched_recurrence_step(
            previous_states=previous_states,
            drivers=x_batched,
        )

        return predicted_states - state_batched

    def compute_residuals(
        self,
        states: torch.Tensor,
        drivers: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return -self.compute_negative_residuals(
            states=states,
            drivers=drivers,
            initial_state=initial_state,
        )

    def compute_jacobians_autograd(
        self,
        states: torch.Tensor,
        drivers: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
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

        flat_prev = previous_states.reshape(-1, self.state_dim)
        flat_drivers = x_batched.reshape(-1, self.input_dim)

        jac_single = jacrev(self.recurrence_step, argnums=0)
        jac_flat = vmap(jac_single)(flat_prev, flat_drivers)

        return jac_flat.reshape(
            x_batched.shape[0],
            x_batched.shape[1],
            self.state_dim,
            self.state_dim,
        )

    def _normalize_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if x.ndim == 2:
            if x.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected unbatched input shape (T, {self.input_dim}), "
                    f"got {tuple(x.shape)}."
                )
            return x.unsqueeze(0), False

        if x.ndim != 3:
            raise ValueError(
                "Input must have shape (T, input_dim), (B, T, input_dim), "
                "or (T, B, input_dim) when batch_first=False."
            )

        if self.batch_first:
            if x.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected batch-first input shape (B, T, {self.input_dim}), "
                    f"got {tuple(x.shape)}."
                )
            return x, True

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected time-first input shape (T, B, {self.input_dim}), "
                f"got {tuple(x.shape)}."
            )

        return x.transpose(0, 1).contiguous(), True

    def _normalize_states(self, states: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if states.ndim == 2:
            if states.shape[-1] != self.state_dim:
                raise ValueError(
                    f"Expected unbatched state shape (T, {self.state_dim}), "
                    f"got {tuple(states.shape)}."
                )
            return states.unsqueeze(0), False

        if states.ndim != 3:
            raise ValueError(
                "States must have shape (T, state_dim), (B, T, state_dim), "
                "or (T, B, state_dim) when batch_first=False."
            )

        if self.batch_first:
            if states.shape[-1] != self.state_dim:
                raise ValueError(
                    f"Expected batch-first states shape (B, T, {self.state_dim}), "
                    f"got {tuple(states.shape)}."
                )
            return states, True

        if states.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected time-first states shape (T, B, {self.state_dim}), "
                f"got {tuple(states.shape)}."
            )

        return states.transpose(0, 1).contiguous(), True

    def _restore_output_layout(
        self,
        y_batched: torch.Tensor,
        had_batch_dim: bool,
    ) -> torch.Tensor:
        if not had_batch_dim:
            return y_batched.squeeze(0)

        if self.batch_first:
            return y_batched

        return y_batched.transpose(0, 1).contiguous()

    def _normalize_initial_state(
        self,
        x_batched: torch.Tensor,
        initial_state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = x_batched.shape[0]
        device = x_batched.device
        dtype = x_batched.dtype

        if initial_state is None:
            return torch.zeros(
                batch_size,
                self.state_dim,
                device=device,
                dtype=dtype,
            )

        if initial_state.ndim == 1:
            if initial_state.shape[0] != self.state_dim:
                raise ValueError(
                    f"Expected initial_state shape ({self.state_dim},), "
                    f"got {tuple(initial_state.shape)}."
                )

            return initial_state.to(
                device=device,
                dtype=dtype,
            ).expand(batch_size, -1)

        if initial_state.ndim == 2:
            if initial_state.shape != (batch_size, self.state_dim):
                raise ValueError(
                    f"Expected initial_state shape ({batch_size}, {self.state_dim}), "
                    f"got {tuple(initial_state.shape)}."
                )

            return initial_state.to(
                device=device,
                dtype=dtype,
            )

        raise ValueError(
            "initial_state must be None, shape (state_dim,), or shape (B, state_dim)."
        )

    def _make_deer_jacobian_fn(self, cfg: DeerNewtonConfig):
        jacobian_backend = getattr(cfg, "jacobian_backend", "autograd")

        if jacobian_backend == "autograd":
            return None

        if jacobian_backend != "explicit":
            raise ValueError(
                "Unknown jacobian_backend "
                f"{jacobian_backend!r}. Expected 'autograd' or 'explicit'."
            )

        if not cfg.quasi:
            raise ValueError(
                "jacobian_backend='explicit' currently provides diagonal "
                "Jacobians, so it must be used with quasi=True. Use "
                "jacobian_backend='autograd' for full DEER."
            )

        if not hasattr(self, "_compute_jacobians_diag_from_previous"):
            raise TypeError(
                "jacobian_backend='explicit' requires the cell to implement "
                "_compute_jacobians_diag_from_previous(previous_states, drivers)."
            )

        def jacobian_fn(
            previous_states: torch.Tensor,
            drivers: torch.Tensor,
        ) -> torch.Tensor:
            if previous_states.ndim != 3:
                raise ValueError(
                    "Expected previous_states with shape (B, T-1, state_dim), got "
                    f"{tuple(previous_states.shape)}."
                )

            if drivers.ndim != 3:
                raise ValueError(
                    "Expected drivers with shape (B, T-1, input_dim), got "
                    f"{tuple(drivers.shape)}."
                )

            return self._compute_jacobians_diag_from_previous(
                previous_states=previous_states,
                drivers=drivers,
            )

        return jacobian_fn

    def _make_deer_linearization_fn(self, cfg: DeerNewtonConfig):
        jacobian_backend = getattr(cfg, "jacobian_backend", "autograd")

        if jacobian_backend == "autograd":
            return None

        if jacobian_backend != "explicit":
            raise ValueError(
                "Unknown jacobian_backend "
                f"{jacobian_backend!r}. Expected 'autograd' or 'explicit'."
            )

        if not cfg.quasi:
            return None

        if not hasattr(self, "compute_linearization_diag_from_previous"):
            return None

        def linearization_fn(
            previous_states: torch.Tensor,
            drivers: torch.Tensor,
        ):
            if previous_states.ndim != 3:
                raise ValueError(
                    "Expected previous_states with shape (B, T, state_dim), got "
                    f"{tuple(previous_states.shape)}."
                )

            if drivers.ndim != 3:
                raise ValueError(
                    "Expected drivers with shape (B, T, input_dim), got "
                    f"{tuple(drivers.shape)}."
                )

            return self.compute_linearization_diag_from_previous(
                previous_states=previous_states,
                drivers=drivers,
            )

        return linearization_fn

    @staticmethod
    def _load_accel_scan_if_needed(cfg: DeerNewtonConfig):
        if cfg.scan_backend != "accel_scan":
            return None

        if cfg.accel_module == "warp":
            from accelerated_scan.warp import scan
            return scan

        if cfg.accel_module == "scalar":
            from accelerated_scan.scalar import scan
            return scan

        if cfg.accel_module == "ref":
            from accelerated_scan.ref import scan
            return scan

        raise ValueError(
            f"Unknown accelerated_scan module: {cfg.accel_module!r}."
        )


BaseDeerRNNCell = BaseParaRNNCell
