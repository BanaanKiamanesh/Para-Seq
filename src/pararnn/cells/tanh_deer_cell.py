from __future__ import annotations

import torch

from src.pararnn.base_cell import BaseDeerRNNCell
from src.pararnn.config import ParaRNNDeerConfig


class TanhDeerRNNCell(BaseDeerRNNCell):
    """Small trainable tanh RNN cell for Phase-1 DEER-backed ParaRNN testing.

    Recurrence:

        h_t = tanh(W_h h_{t-1} + W_x x_t + b).

    This is not the final ParaGRU cell. It is the minimal correctness target for
    the new src/pararnn subsystem while reusing src.algos.DEER.deer_alg.
    """

    def __init__(self, config: ParaRNNDeerConfig):
        super().__init__(config)

        self.W_h = torch.nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.W_x = torch.nn.Linear(self.input_dim, self.state_dim, bias=True)

        self.reset_parameters()

        if config.device is not None or config.dtype is not None:
            self.to(device=config.device, dtype=config.dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.W_h.weight, mean=0.0, std=0.25)
        torch.nn.init.normal_(self.W_x.weight, mean=0.0, std=0.25)
        torch.nn.init.zeros_(self.W_x.bias)

        # Keep the first toy cell contractive enough that Newton converges fast.
        with torch.no_grad():
            self.W_h.weight.clamp_(-0.5, 0.5)

    def recurrence_step(self, state: torch.Tensor, driver: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.W_h(state) + self.W_x(driver))
