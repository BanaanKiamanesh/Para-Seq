import torch


ACCEL_SCAN_MIN_LEN = 32
ACCEL_SCAN_MAX_LEN = 65536


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def validate_accel_scan_inputs(
    A: torch.Tensor,
    b: torch.Tensor,
    expected_ndim: int,
) -> None:
    if A.ndim != expected_ndim or b.ndim != expected_ndim:
        if expected_ndim == 2:
            expected_shape = "(T, D)"
        elif expected_ndim == 3:
            expected_shape = "(B, T, D)"
        else:
            expected_shape = f"{expected_ndim}-dimensional tensors"

        raise ValueError(f"Expected A and b with shape {expected_shape}.")

    if A.shape != b.shape:
        raise ValueError(
            f"A and b must have the same shape, got {A.shape} and {b.shape}."
        )

    if A.device.type != "cuda":
        raise ValueError("accelerated_scan backend requires CUDA tensors.")


def run_accel_scan_chunk_batched(
    A_chunk: torch.Tensor,
    b_chunk: torch.Tensor,
    accel_scan_fn,
    min_len: int = ACCEL_SCAN_MIN_LEN,
    max_len: int = ACCEL_SCAN_MAX_LEN,
) -> torch.Tensor:
    """Run accelerated_scan on one batched diagonal affine chunk.

    accelerated_scan expects tensors in (B, D, T) layout. The repository uses
    (B, T, D), so this helper handles layout conversion, padding, and validation.
    """
    if accel_scan_fn is None:
        raise ValueError(
            "accel_scan_fn must be provided when scan_backend='accel_scan'."
        )

    if A_chunk.ndim != 3 or b_chunk.ndim != 3:
        raise ValueError(
            "Expected A_chunk and b_chunk with shape (B, T, D), got "
            f"{tuple(A_chunk.shape)} and {tuple(b_chunk.shape)}."
        )

    if A_chunk.shape != b_chunk.shape:
        raise ValueError(
            "A_chunk and b_chunk must have the same shape, got "
            f"{tuple(A_chunk.shape)} and {tuple(b_chunk.shape)}."
        )

    batch_size, original_len, state_dim = A_chunk.shape
    padded_len = next_power_of_two(max(original_len, min_len))

    if padded_len > max_len:
        raise ValueError(
            f"accelerated_scan chunk length must be <= {max_len}, "
            f"but got padded_len={padded_len}."
        )

    if padded_len != original_len:
        pad_len = padded_len - original_len

        A_pad = torch.ones(
            batch_size,
            pad_len,
            state_dim,
            device=A_chunk.device,
            dtype=A_chunk.dtype,
        )
        b_pad = torch.zeros(
            batch_size,
            pad_len,
            state_dim,
            device=b_chunk.device,
            dtype=b_chunk.dtype,
        )

        A_chunk = torch.cat([A_chunk, A_pad], dim=1)
        b_chunk = torch.cat([b_chunk, b_pad], dim=1)

    gate = A_chunk.transpose(1, 2).contiguous()
    token = b_chunk.transpose(1, 2).contiguous()

    scanned = accel_scan_fn(gate, token)
    b_prefix = scanned.transpose(1, 2).contiguous()

    return b_prefix[:, :original_len, :]


def diag_mat_scan_accel_batched(
    A: torch.Tensor,
    b: torch.Tensor,
    accel_scan_fn,
    max_len: int = ACCEL_SCAN_MAX_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched accelerated-scan backend for diagonal affine recurrences.

    Solves

        h_{b,t} = A_{b,t} * h_{b,t-1} + b_{b,t}

    for all batch items and hidden coordinates using one accelerated_scan call
    per chunk.
    """
    validate_accel_scan_inputs(A, b, expected_ndim=3)

    batch_size, seq_len, state_dim = A.shape

    if seq_len == 0:
        return A.clone(), b.clone()

    b_outputs = []
    A_outputs = []

    state_carry = torch.zeros(
        batch_size,
        state_dim,
        device=A.device,
        dtype=A.dtype,
    )
    A_carry = torch.ones(
        batch_size,
        state_dim,
        device=A.device,
        dtype=A.dtype,
    )

    start = 0

    while start < seq_len:
        end = min(start + max_len, seq_len)

        A_chunk = A[:, start:end, :].contiguous()
        b_chunk = b[:, start:end, :].contiguous()

        b_prefix_zero = run_accel_scan_chunk_batched(
            A_chunk=A_chunk,
            b_chunk=b_chunk,
            accel_scan_fn=accel_scan_fn,
            max_len=max_len,
        )

        A_prefix_local = torch.cumprod(A_chunk, dim=1)

        b_prefix = A_prefix_local * state_carry[:, None, :] + b_prefix_zero
        A_prefix = A_prefix_local * A_carry[:, None, :]

        state_carry = b_prefix[:, -1, :]
        A_carry = A_prefix[:, -1, :]

        b_outputs.append(b_prefix)
        A_outputs.append(A_prefix)

        start = end

    return torch.cat(A_outputs, dim=1), torch.cat(b_outputs, dim=1)


def run_accel_scan_chunk(
    A_chunk: torch.Tensor,
    b_chunk: torch.Tensor,
    accel_scan_fn,
    min_len: int = ACCEL_SCAN_MIN_LEN,
    max_len: int = ACCEL_SCAN_MAX_LEN,
) -> torch.Tensor:
    """Unbatched compatibility wrapper for one accelerated scan chunk."""
    if A_chunk.ndim != 2 or b_chunk.ndim != 2:
        raise ValueError(
            "Expected A_chunk and b_chunk with shape (T, D), got "
            f"{tuple(A_chunk.shape)} and {tuple(b_chunk.shape)}."
        )

    return run_accel_scan_chunk_batched(
        A_chunk=A_chunk.unsqueeze(0),
        b_chunk=b_chunk.unsqueeze(0),
        accel_scan_fn=accel_scan_fn,
        min_len=min_len,
        max_len=max_len,
    ).squeeze(0)


def diag_mat_scan_accel(
    A: torch.Tensor,
    b: torch.Tensor,
    accel_scan_fn,
    max_len: int = ACCEL_SCAN_MAX_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unbatched accelerated-scan backend for diagonal affine recurrences."""
    validate_accel_scan_inputs(A, b, expected_ndim=2)

    A_prefix, b_prefix = diag_mat_scan_accel_batched(
        A=A.unsqueeze(0),
        b=b.unsqueeze(0),
        accel_scan_fn=accel_scan_fn,
        max_len=max_len,
    )

    return A_prefix.squeeze(0), b_prefix.squeeze(0)
