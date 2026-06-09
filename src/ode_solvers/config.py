from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


IntegratorMethod = Literal["euler", "midpoint", "heun", "rk2", "rk4"]
ODESolverName = Literal["sequential", "deer", "elk"]
ScanBackend = Literal["torch", "accel_scan"]
StoppingCriterion = Literal["update", "merit"]
InitialGuess = Literal["zero", "constant", "f0"]


@dataclass
class ODESolverConfig:
    method: IntegratorMethod = "rk4"
    solver: ODESolverName = "deer"
    num_iters: int = 20
    tol: Optional[float] = None
    strict_tol: bool = False
    stopping_criterion: StoppingCriterion = "update"
    initial_guess: InitialGuess = "f0"
    quasi: bool = True
    damping: float = 0.0
    clip_value: Optional[float] = None
    scan_backend: ScanBackend = "torch"
    accel_module: str = "warp"
    sigmasq: float = 1e8
    process_noise: float = 1.0
    include_initial: bool = True
