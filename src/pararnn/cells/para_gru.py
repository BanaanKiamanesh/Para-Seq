from __future__ import annotations

from dataclasses import dataclass

import torch

from src.pararnn.base_cell import BaseParaRNNCell
from src.pararnn.config import ParaRNNConfig


@dataclass
class ParaGRUConfig(ParaRNNConfig):
    """Configuration for the simple diagonal ParaGRU cell.

    This is the first real ParaRNN-style cell in this repository.

    For now we implement the non-multi-head version:

        A: (3, state_dim)
        B: (3, input_dim, state_dim)
        b: (3, state_dim)

    Later we will upgrade this to the multi-head ParaRNN layout:

        B: (num_heads, head_input_dim, 3, head_state_dim)
    """

    recurrent_init_scale: float = 0.25
    input_init_scale: float = 1.0
    bias_init_value: float = 0.0


class ParaGRUCell(BaseParaRNNCell):
    """Diagonal ParaGRU cell.

    Recurrence:

        z_t = sigmoid(a_z * h_{t-1} + B_z x_t + b_z)

        r_t = sigmoid(a_r * h_{t-1} + B_r x_t + b_r)

        c_t = tanh(a_h * (h_{t-1} * r_t) + B_h x_t + b_h)

        h_t = z_t * c_t + (1 - z_t) * h_{t-1}

    The recurrent matrices are diagonal. Therefore the Jacobian

        d h_t / d h_{t-1}

    is diagonal and can be represented by a tensor of shape (..., state_dim).
    """

    def __init__(self, config: ParaGRUConfig | ParaRNNConfig):
        super().__init__(config)

        if self.output_dim != self.state_dim:
            raise ValueError(
                "ParaGRUCell uses identity post_process, so output_dim must equal state_dim."
            )

        self.A = torch.nn.Parameter(torch.empty(3, self.state_dim))
        self.B = torch.nn.Parameter(torch.empty(
            3, self.input_dim, self.state_dim))
        self.b = torch.nn.Parameter(torch.empty(3, self.state_dim))

        self.reset_parameters()

        if config.device is not None or config.dtype is not None:
            self.to(device=config.device, dtype=config.dtype)

    def reset_parameters(self) -> None:
        recurrent_init_scale = float(
            getattr(self.config, "recurrent_init_scale", 0.25)
        )
        bias_init_value = float(
            getattr(self.config, "bias_init_value", 0.0)
        )

        torch.nn.init.uniform_(
            self.A,
            a=-recurrent_init_scale,
            b=recurrent_init_scale,
        )

        for gate_idx in range(3):
            torch.nn.init.xavier_uniform_(self.B[gate_idx])

        torch.nn.init.constant_(self.b, bias_init_value)

    def recurrence_step(
        self,
        state: torch.Tensor,
        driver: torch.Tensor,
    ) -> torch.Tensor:
        """One ParaGRU recurrent update.

        Args:
            state:
                Previous hidden state with shape (..., state_dim).

            driver:
                Input with shape (..., input_dim).

        Returns:
            Next hidden state with shape (..., state_dim).
        """
        Bx_plus_b = self._input_projection(driver)

        z_pre = self.A[0] * state + Bx_plus_b[..., 0, :]
        r_pre = self.A[1] * state + Bx_plus_b[..., 1, :]

        z = torch.sigmoid(z_pre)
        r = torch.sigmoid(r_pre)

        c_pre = self.A[2] * (state * r) + Bx_plus_b[..., 2, :]
        c = torch.tanh(c_pre)

        next_state = z * c + (1.0 - z) * state

        return next_state

    def assemble_initial_guess(
        self,
        drivers: torch.Tensor,
        initial_state: torch.Tensor,
        guess_type: str = "f0",
    ) -> torch.Tensor:
        """Assemble the Newton initial guess for one sequence.

        For ``guess_type='f0'`` this computes

            h_t^0 = f(0, x_t)

        in vectorized form across time.
        """
        if guess_type == "zero":
            return torch.zeros(
                drivers.shape[0],
                self.state_dim,
                device=drivers.device,
                dtype=drivers.dtype,
            )

        if guess_type == "f0":
            zero_states = torch.zeros(
                drivers.shape[0],
                self.state_dim,
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
        """Compute the explicit diagonal ParaGRU Jacobian.

        Returns:
            Tensor with shape (B, T, state_dim), representing the diagonal of

                d f(h_{t-1}, x_t) / d h_{t-1}.

        For an unbatched input, the returned tensor still uses the normalized
        batched shape (1, T, state_dim). This matches the base class Jacobian
        convention.
        """
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
        """Compute diagonals for the reverse-time backward recurrence.

        This prepares the same object used later for a ParaRNN-style custom
        backward pass. For a diagonal Jacobian, transposition does not change
        entries, but the recurrence must be flipped in time.
        """
        jac = self.compute_jacobians_diag(
            states=states,
            drivers=drivers,
            initial_state=initial_state,
        )

        jac_bwd = torch.roll(
            torch.flip(jac, dims=[1]),
            shifts=1,
            dims=1,
        )

        jac_bwd[:, 0, :] = 0.0

        return jac_bwd

    def _input_projection(self, driver: torch.Tensor) -> torch.Tensor:
        """Compute all three input projections B_g x + b_g.

        Args:
            driver:
                Tensor with shape (..., input_dim).

        Returns:
            Tensor with shape (..., 3, state_dim).
        """
        return torch.einsum("...i,gij->...gj", driver, self.B) + self.b

    def compute_linearization_diag_from_previous(
        self,
        previous_states: torch.Tensor,
        drivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ParaGRU recurrence values and diagonal Jacobians together.

        This is the fused linearization used by explicit quasi-DEER. It avoids
        computing the ParaGRU gates once for ``f(h_{t-1}, x_t)`` and a second
        time for ``diag(df/dh)``.

        Args:
            previous_states:
                Tensor with shape ``(B, T, state_dim)``, containing
                ``h_{t-1}``.

            drivers:
                Tensor with shape ``(B, T, input_dim)``, containing ``x_t``.

        Returns:
            predicted_states:
                Tensor with shape ``(B, T, state_dim)``, containing
                ``f(h_{t-1}, x_t)``.

            jacobian_diag:
                Tensor with shape ``(B, T, state_dim)``, containing the diagonal
                entries of ``df(h_{t-1}, x_t) / dh_{t-1}``.
        """
        if previous_states.ndim != 3:
            raise ValueError(
                "previous_states must have shape (B, T, state_dim), "
                f"got {tuple(previous_states.shape)}."
            )

        if drivers.ndim != 3:
            raise ValueError(
                "drivers must have shape (B, T, input_dim), "
                f"got {tuple(drivers.shape)}."
            )

        if previous_states.shape[:2] != drivers.shape[:2]:
            raise ValueError(
                "previous_states and drivers must share batch/time dimensions, "
                f"got {tuple(previous_states.shape)} and {tuple(drivers.shape)}."
            )

        h_prev = previous_states

        Bx_plus_b = self._input_projection(drivers)

        z_pre = self.A[0] * h_prev + Bx_plus_b[..., 0, :]
        r_pre = self.A[1] * h_prev + Bx_plus_b[..., 1, :]

        z = torch.sigmoid(z_pre)
        r = torch.sigmoid(r_pre)

        c_pre = self.A[2] * (h_prev * r) + Bx_plus_b[..., 2, :]
        c = torch.tanh(c_pre)

        predicted_states = z * c + (1.0 - z) * h_prev

        dz_dpre = z * (1.0 - z)
        dr_dpre = r * (1.0 - r)
        dc_dpre = 1.0 - c * c

        dz_dh = self.A[0] * dz_dpre
        dr_dh = self.A[1] * dr_dpre

        dcpre_dh = self.A[2] * (r + h_prev * dr_dh)
        dc_dh = dc_dpre * dcpre_dh

        jacobian_diag = (
            (1.0 - z)
            + (c - h_prev) * dz_dh
            + z * dc_dh
        )

        return predicted_states, jacobian_diag

    def _compute_jacobians_diag_from_previous(
        self,
        previous_states: torch.Tensor,
        drivers: torch.Tensor,
    ) -> torch.Tensor:
        """Compute explicit diagonal Jacobians from previous states.

        Args:
            previous_states:
                Tensor with shape (B, T, state_dim), containing h_{t-1}.

            drivers:
                Tensor with shape (B, T, input_dim), containing x_t.

        Returns:
            Tensor with shape (B, T, state_dim).
        """
        _, jacobian_diag = self.compute_linearization_diag_from_previous(
            previous_states=previous_states,
            drivers=drivers,
        )

        return jacobian_diag


# Shorter alias for convenience.
ParaGRU = ParaGRUCell
