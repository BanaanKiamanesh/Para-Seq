from src.algos.DEER import (
    deer_alg,
    deer_alg_batched,
    deer_step,
    deer_step_batched,
    sequential_rollout,
    get_residual,
    get_residual_batched,
    merit_fxn,
    merit_fxn_batched,
)

from src.algos.Picard import (
    picard_alg,
    picard_step,
)

from src.algos.Jacobi import (
    jacobi_alg,
    jacobi_step,
)

from src.algos.ELK import (
    elk_alg,
    elk_step,
)


# === Unified algorithm mode exports ===
from src.algos.FixedPoint import (
    fixed_point_step_batched,
    fixed_point_alg_batched,
    jacobi_alg_batched,
    picard_alg_batched,
)

try:
    __all__
except NameError:
    __all__ = []

for _name in [
    "fixed_point_step_batched",
    "fixed_point_alg_batched",
    "jacobi_alg_batched",
    "picard_alg_batched",
]:
    if _name not in __all__:
        __all__.append(_name)
# === End unified algorithm mode exports ===
