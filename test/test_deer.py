import time

import torch

from src.algos.DEER import deer_alg, sequential_rollout
from src.algos.Picard import picard_alg
from src.algos.Jacobi import jacobi_alg


class SimpleRNNCell(torch.nn.Module):
    def __init__(self, state_dim, input_dim):
        super().__init__()

        self.W_h = torch.nn.Linear(state_dim, state_dim, bias=False)
        self.W_u = torch.nn.Linear(input_dim, state_dim, bias=True)

        torch.nn.init.normal_(self.W_h.weight, mean=0.0, std=0.25)
        torch.nn.init.normal_(self.W_u.weight, mean=0.0, std=0.25)
        torch.nn.init.zeros_(self.W_u.bias)

    def forward(self, state, driver):
        return torch.tanh(self.W_h(state) + self.W_u(driver))


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_function(fn, device):
    sync_if_cuda(device)

    start_time = time.perf_counter()
    result = fn()

    sync_if_cuda(device)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return result, elapsed_time


def print_report(name, states, info, elapsed_time, true_states):
    error = torch.max(torch.abs(states - true_states))

    print(f"\n{name}")
    print("iters:", info["num_iters"])
    print("time:", elapsed_time)
    print("initial merit:", info["initial_merit"].item())
    print("final merit:", info["final_merit"].item())
    print("max error vs sequential:", error.item())


def main():
    torch.manual_seed(0)

    device = torch.device("cuda")
    dtype = torch.float64

    T = 1024 * 128
    state_dim = 4
    input_dim = 3

    cell = SimpleRNNCell(state_dim, input_dim).to(device=device, dtype=dtype)

    initial_state = torch.zeros(state_dim, device=device, dtype=dtype)
    drivers = torch.randn(T, input_dim, device=device, dtype=dtype)

    def f(state, driver):
        return cell(state, driver)

    # ------------------------------------------------------------
    # Sequential evaluation timing
    # ------------------------------------------------------------
    true_states, sequential_time = time_function(
        lambda: sequential_rollout(f, initial_state, drivers),
        device=device,
    )

    states_guess = torch.zeros_like(true_states)

    # ------------------------------------------------------------
    # Full DEER timing
    # ------------------------------------------------------------
    (deer_states, deer_info), deer_time = time_function(
        lambda: deer_alg(
            f=f,
            initial_state=initial_state,
            states_guess=states_guess,
            drivers=drivers,
            num_iters=T,
            tol=1e-12,
            quasi=False,
            damping=0.0,
            clip_value=1e8,
            return_trace=False,
        ),
        device=device,
    )

    # ------------------------------------------------------------
    # Quasi-DEER timing
    # ------------------------------------------------------------
    (quasi_states, quasi_info), quasi_time = time_function(
        lambda: deer_alg(
            f=f,
            initial_state=initial_state,
            states_guess=states_guess,
            drivers=drivers,
            num_iters=T,
            tol=1e-12,
            quasi=True,
            damping=0.0,
            clip_value=1e8,
            return_trace=False,
        ),
        device=device,
    )

    # ------------------------------------------------------------
    # Picard timing
    # ------------------------------------------------------------
    # Picard is cheaper per iteration than DEER, but can need many more iterations.
    picard_max_iters = 256

    (picard_states, picard_info), picard_time = time_function(
        lambda: picard_alg(
            f=f,
            initial_state=initial_state,
            states_guess=states_guess,
            drivers=drivers,
            num_iters=picard_max_iters,
            tol=1e-12,
            clip_value=1e8,
            return_trace=False,
        ),
        device=device,
    )

    # ------------------------------------------------------------
    # Jacobi timing
    # ------------------------------------------------------------
    # Jacobi can require up to T iterations in the worst case.
    # For T = 65536, do not start with num_iters=T unless you want a very long run.
    jacobi_max_iters = 256

    (jacobi_states, jacobi_info), jacobi_time = time_function(
        lambda: jacobi_alg(
            f=f,
            initial_state=initial_state,
            states_guess=states_guess,
            drivers=drivers,
            num_iters=jacobi_max_iters,
            tol=1e-12,
            clip_value=1e8,
            return_trace=False,
        ),
        device=device,
    )

    print("Device:", device)

    print("\nSequential Evaluation")
    print("time:", sequential_time)
    print("final merit:", 0.0)
    print("max error vs sequential:", 0.0)

    print_report("Full DEER", deer_states, deer_info, deer_time, true_states)
    print_report("Quasi-DEER", quasi_states,
                 quasi_info, quasi_time, true_states)
    print_report("Picard", picard_states, picard_info,
                 picard_time, true_states)
    print_report("Jacobi", jacobi_states, jacobi_info,
                 jacobi_time, true_states)


if __name__ == "__main__":
    main()
