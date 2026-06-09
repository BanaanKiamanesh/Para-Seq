from __future__ import annotations

import torch

from src.utils.AssScan import full_mat_scan, diag_mat_scan
from src.utils.AccelScan import diag_mat_scan_accel_batched


def _load_accel_scan(accel_module: str):
    if accel_module == "warp":
        from accelerated_scan.warp import scan
        return scan

    if accel_module == "scalar":
        from accelerated_scan.scalar import scan
        return scan

    if accel_module == "ref":
        from accelerated_scan.ref import scan
        return scan

    raise ValueError(f"Unknown accelerated_scan module: {accel_module!r}.")


def _normalize_u(U: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if U.ndim == 2:
        return U.unsqueeze(0), False

    if U.ndim == 3:
        return U, True

    raise ValueError(
        "U must have shape (T, input_dim) or (B, T, input_dim), "
        f"got {tuple(U.shape)}."
    )


def _normalize_x0(
    x0: torch.Tensor | None,
    batch_size: int,
    state_dim: int,
    *,
    device,
    dtype,
) -> torch.Tensor:
    if x0 is None:
        return torch.zeros(batch_size, state_dim, device=device, dtype=dtype)

    x0 = torch.as_tensor(x0, device=device, dtype=dtype)

    if x0.ndim == 1:
        if x0.shape[0] != state_dim:
            raise ValueError(
                f"Expected x0 shape ({state_dim},), got {tuple(x0.shape)}."
            )
        return x0.unsqueeze(0).expand(batch_size, -1)

    if x0.ndim == 2:
        expected = (batch_size, state_dim)
        if tuple(x0.shape) != expected:
            raise ValueError(f"Expected x0 shape {expected}, got {tuple(x0.shape)}.")
        return x0

    raise ValueError("x0 must be None, shape (state_dim,), or shape (B, state_dim).")


def _infer_fixed_dt(t: torch.Tensor | None, dt: float | None, num_samples: int) -> float:
    if dt is not None:
        return float(dt)

    if t is None:
        raise ValueError("Either t or dt must be provided.")

    if t.ndim != 1:
        raise ValueError(f"t must have shape (T,), got {tuple(t.shape)}.")

    if t.shape[0] != num_samples:
        raise ValueError(
            f"t must have the same length as U over time. Got len(t)={t.shape[0]} "
            f"and U time length={num_samples}."
        )

    if num_samples < 2:
        raise ValueError("At least two time samples are needed to infer dt.")

    dts = t[1:] - t[:-1]
    dt0 = dts[0]

    if not torch.allclose(dts, dt0.expand_as(dts), rtol=1e-5, atol=1e-8):
        raise ValueError("Only fixed-step time grids are supported.")

    return float(dt0.item())


def discretize_lti_zoh(
    A: torch.Tensor,
    B: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must have shape (D, D), got {tuple(A.shape)}.")

    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError(
            f"B must have shape (D, M) with D={A.shape[0]}, got {tuple(B.shape)}."
        )

    state_dim = A.shape[0]
    input_dim = B.shape[1]

    upper = torch.cat([A, B], dim=1)
    lower = torch.zeros(
        input_dim,
        state_dim + input_dim,
        device=A.device,
        dtype=A.dtype,
    )
    augmented = torch.cat([upper, lower], dim=0)

    exp_augmented = torch.matrix_exp(augmented * dt)

    Phi = exp_augmented[:state_dim, :state_dim]
    Gamma = exp_augmented[:state_dim, state_dim:]

    return Phi, Gamma


def discretize_lti_zoh_diag(
    A_diag: torch.Tensor,
    B: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if A_diag.ndim == 2:
        if A_diag.shape[0] != A_diag.shape[1]:
            raise ValueError(
                f"A_diag matrix must have shape (D, D), got {tuple(A_diag.shape)}."
            )
        A_diag = torch.diagonal(A_diag, dim1=-2, dim2=-1)

    if A_diag.ndim != 1:
        raise ValueError(
            "A_diag must have shape (D,) or be a diagonal matrix with shape (D, D)."
        )

    if B.ndim != 2 or B.shape[0] != A_diag.shape[0]:
        raise ValueError(
            f"B must have shape (D, M) with D={A_diag.shape[0]}, got {tuple(B.shape)}."
        )

    Phi_diag = torch.exp(A_diag * dt)

    denom = A_diag
    small = torch.abs(denom) <= torch.finfo(A_diag.dtype).eps
    scale = torch.where(
        small,
        torch.full_like(denom, float(dt)),
        (Phi_diag - 1.0) / denom,
    )

    Gamma = scale[:, None] * B

    return Phi_diag, Gamma


def linear_affine_scan(
    Phi: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
    *,
    diagonal: bool = False,
    scan_backend: str = "torch",
    accel_scan_fn=None,
    accel_module: str = "warp",
) -> torch.Tensor:
    if scan_backend not in ("torch", "accel_scan"):
        raise ValueError("scan_backend must be 'torch' or 'accel_scan'.")

    b_batched, had_batch_dim = _normalize_u(b)
    batch_size, num_steps, state_dim = b_batched.shape

    x0_batched = _normalize_x0(
        x0,
        batch_size,
        state_dim,
        device=b_batched.device,
        dtype=b_batched.dtype,
    )

    if num_steps == 0:
        out = torch.empty(
            batch_size,
            0,
            state_dim,
            device=b_batched.device,
            dtype=b_batched.dtype,
        )
        return out if had_batch_dim else out.squeeze(0)

    if diagonal:
        if Phi.ndim == 2:
            Phi_diag = torch.diagonal(Phi, dim1=-2, dim2=-1)
        elif Phi.ndim == 1:
            Phi_diag = Phi
        else:
            raise ValueError(
                "For diagonal=True, Phi must have shape (D,) or (D, D)."
            )

        Phi_diag = Phi_diag.to(device=b_batched.device, dtype=b_batched.dtype)
        A_terms = Phi_diag.expand(batch_size, num_steps, state_dim)

        if scan_backend == "torch":
            A_prefix, b_prefix = diag_mat_scan(A_terms, b_batched, dim=1)
        else:
            if accel_scan_fn is None:
                accel_scan_fn = _load_accel_scan(accel_module)
            A_prefix, b_prefix = diag_mat_scan_accel_batched(
                A=A_terms,
                b=b_batched,
                accel_scan_fn=accel_scan_fn,
            )

        states = A_prefix * x0_batched[:, None, :] + b_prefix
        return states if had_batch_dim else states.squeeze(0)

    if scan_backend != "torch":
        raise ValueError("Dense linear scans only support scan_backend='torch'.")

    if Phi.ndim != 2 or Phi.shape != (state_dim, state_dim):
        raise ValueError(
            f"Dense Phi must have shape ({state_dim}, {state_dim}), got {tuple(Phi.shape)}."
        )

    Phi = Phi.to(device=b_batched.device, dtype=b_batched.dtype)
    A_terms = Phi.expand(batch_size, num_steps, state_dim, state_dim)

    A_prefix, b_prefix = full_mat_scan(A_terms, b_batched, dim=1)
    states = (A_prefix @ x0_batched[:, None, :, None]).squeeze(-1) + b_prefix

    return states if had_batch_dim else states.squeeze(0)


def dlsim(
    Phi: torch.Tensor,
    Gamma: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    U: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    diagonal: bool = False,
    scan_backend: str = "torch",
    accel_scan_fn=None,
    accel_module: str = "warp",
    return_x: bool = True,
):
    Phi = torch.as_tensor(Phi)
    Gamma = torch.as_tensor(Gamma, device=Phi.device, dtype=Phi.dtype)
    C = torch.as_tensor(C, device=Phi.device, dtype=Phi.dtype)
    D = torch.as_tensor(D, device=Phi.device, dtype=Phi.dtype)
    U = torch.as_tensor(U, device=Phi.device, dtype=Phi.dtype)

    U_batched, had_batch_dim = _normalize_u(U)
    batch_size, num_samples, input_dim = U_batched.shape

    if diagonal:
        state_dim = Phi.shape[0] if Phi.ndim == 1 else Phi.shape[0]
    else:
        state_dim = Phi.shape[0]

    if Gamma.shape != (state_dim, input_dim):
        raise ValueError(
            f"Gamma must have shape ({state_dim}, {input_dim}), got {tuple(Gamma.shape)}."
        )

    if C.ndim != 2 or C.shape[1] != state_dim:
        raise ValueError(f"C must have shape (output_dim, {state_dim}).")

    if D.ndim != 2 or D.shape[1] != input_dim or D.shape[0] != C.shape[0]:
        raise ValueError(
            f"D must have shape ({C.shape[0]}, {input_dim}), got {tuple(D.shape)}."
        )

    x0_batched = _normalize_x0(
        x0,
        batch_size,
        state_dim,
        device=U_batched.device,
        dtype=U_batched.dtype,
    )

    if num_samples == 0:
        raise ValueError("U must contain at least one time sample.")

    if num_samples == 1:
        x_full = x0_batched[:, None, :]
    else:
        b_terms = torch.einsum("btm,dm->btd", U_batched[:, :-1, :], Gamma)
        x_tail = linear_affine_scan(
            Phi=Phi.to(device=U_batched.device, dtype=U_batched.dtype),
            b=b_terms,
            x0=x0_batched,
            diagonal=diagonal,
            scan_backend=scan_backend,
            accel_scan_fn=accel_scan_fn,
            accel_module=accel_module,
        )
        if x_tail.ndim == 2:
            x_tail = x_tail.unsqueeze(0)
        x_full = torch.cat([x0_batched[:, None, :], x_tail], dim=1)

    y = torch.einsum("od,btd->bto", C, x_full) + torch.einsum(
        "om,btm->bto", D, U_batched
    )

    if not had_batch_dim:
        y = y.squeeze(0)
        x_full = x_full.squeeze(0)

    if return_x:
        return y, x_full

    return y


def lsim(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    U: torch.Tensor,
    t: torch.Tensor | None = None,
    *,
    dt: float | None = None,
    x0: torch.Tensor | None = None,
    diagonal: bool = False,
    scan_backend: str = "torch",
    accel_scan_fn=None,
    accel_module: str = "warp",
    return_x: bool = True,
):
    A = torch.as_tensor(A)
    B = torch.as_tensor(B, device=A.device, dtype=A.dtype)
    C = torch.as_tensor(C, device=A.device, dtype=A.dtype)
    D = torch.as_tensor(D, device=A.device, dtype=A.dtype)
    U = torch.as_tensor(U, device=A.device, dtype=A.dtype)

    U_batched, _ = _normalize_u(U)

    if t is not None:
        t = torch.as_tensor(t, device=A.device, dtype=A.dtype)

    fixed_dt = _infer_fixed_dt(t, dt, U_batched.shape[1])

    if diagonal:
        Phi, Gamma = discretize_lti_zoh_diag(A, B, fixed_dt)
    else:
        Phi, Gamma = discretize_lti_zoh(A, B, fixed_dt)

    return dlsim(
        Phi=Phi,
        Gamma=Gamma,
        C=C,
        D=D,
        U=U,
        x0=x0,
        diagonal=diagonal,
        scan_backend=scan_backend,
        accel_scan_fn=accel_scan_fn,
        accel_module=accel_module,
        return_x=return_x,
    )
