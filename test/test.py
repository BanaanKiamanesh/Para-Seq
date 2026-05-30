import gc
import time

import torch

from src.algos.DEER import deer_alg, sequential_rollout
from src.algos.Picard import picard_alg
from src.algos.Jacobi import jacobi_alg
from src.algos.ELK import elk_alg


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


def clear_cuda_memory(device):
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def time_function(fn, device):
    sync_if_cuda(device)

    start_time = time.perf_counter()
    result = fn()

    sync_if_cuda(device)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return result, elapsed_time


def make_states_guess(T, state_dim, device, dtype):
    return torch.zeros(T, state_dim, device=device, dtype=dtype)


def print_method_report(
    name,
    states,
    info,
    elapsed_time,
    true_states_cpu,
    device,
):
    states_cpu = states.detach().cpu()
    error = torch.max(torch.abs(states_cpu - true_states_cpu)).item()

    initial_merit = info["initial_merit"].detach().cpu().item()
    final_merit = info["final_merit"].detach().cpu().item()

    print(f"\n{name}")
    print("iters:", info["num_iters"])
    print("time:", elapsed_time)
    print("initial merit:", initial_merit)
    print("final merit:", final_merit)
    print("max error vs sequential:", error)

    if "sigmasq" in info:
        print("sigmasq:", info["sigmasq"])

    if "process_noise" in info:
        print("process_noise:", info["process_noise"])

    del states_cpu
    clear_cuda_memory(device)


def run_and_cleanup(
    name,
    fn,
    device,
    true_states_cpu,
):
    result, elapsed_time = time_function(fn, device=device)

    states, info = result

    print_method_report(
        name=name,
        states=states,
        info=info,
        elapsed_time=elapsed_time,
        true_states_cpu=true_states_cpu,
        device=device,
    )

    del states
    del info
    del result

    clear_cuda_memory(device)


def main():
    torch.manual_seed(0)

    device = torch.device("cuda")
    dtype = torch.float64

    T = 1024 * 64
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

    true_states_cpu = true_states.detach().cpu()

    print("Device:", device)

    print("\nSequential Evaluation")
    print("time:", sequential_time)
    print("final merit:", 0.0)
    print("max error vs sequential:", 0.0)

    del true_states
    clear_cuda_memory(device)

    # ------------------------------------------------------------
    # Full DEER / Newton timing
    # ------------------------------------------------------------
    run_and_cleanup(
        name="Full DEER / Newton",
        fn=lambda: deer_alg(
            f=f,
            initial_state=initial_state,
            states_guess=make_states_guess(T, state_dim, device, dtype),
            drivers=drivers,
            num_iters=T,
            tol=1e-12,
            quasi=False,
            damping=0.0,
            clip_value=1e8,
            return_trace=False,
        ),
        device=device,
        true_states_cpu=true_states_cpu,
    )

    # ------------------------------------------------------------
    # Quasi-DEER / Quasi-Newton timing
    # ------------------------------------------------------------
    run_and_cleanup(
        name="Quasi-DEER / Quasi-Newton",
        fn=lambda: deer_alg(
            f=f,
            initial_state=initial_state,
            states_guess=make_states_guess(T, state_dim, device, dtype),
            drivers=drivers,
            num_iters=T,
            tol=1e-12,
            quasi=True,
            damping=0.0,
            clip_value=1e8,
            return_trace=False,
        ),
        device=device,
        true_states_cpu=true_states_cpu,
    )

    # ------------------------------------------------------------
    # Picard timing
    # ------------------------------------------------------------
    picard_max_iters = 256

    run_and_cleanup(
        name="Picard",
        fn=lambda: picard_alg(
            f=f,
            initial_state=initial_state,
            states_guess=make_states_guess(T, state_dim, device, dtype),
            drivers=drivers,
            num_iters=picard_max_iters,
            tol=1e-12,
            clip_value=1e8,
            return_trace=False,
        ),
        device=device,
        true_states_cpu=true_states_cpu,
    )

    # ------------------------------------------------------------
    # Jacobi timing
    # ------------------------------------------------------------
    jacobi_max_iters = 256

    run_and_cleanup(
        name="Jacobi",
        fn=lambda: jacobi_alg(
            f=f,
            initial_state=initial_state,
            states_guess=make_states_guess(T, state_dim, device, dtype),
            drivers=drivers,
            num_iters=jacobi_max_iters,
            tol=1e-12,
            clip_value=1e8,
            return_trace=False,
        ),
        device=device,
        true_states_cpu=true_states_cpu,
    )

    # ------------------------------------------------------------
    # ELK timing
    # ------------------------------------------------------------
    elk_max_iters = 64

    run_and_cleanup(
        name="ELK",
        fn=lambda: elk_alg(
            f=f,
            initial_state=initial_state,
            states_guess=make_states_guess(T, state_dim, device, dtype),
            drivers=drivers,
            sigmasq=1e8,
            process_noise=1.0,
            num_iters=elk_max_iters,
            tol=1e-12,
            quasi=False,
            damping=0.0,
            clip_value=1e8,
            return_trace=False,
        ),
        device=device,
        true_states_cpu=true_states_cpu,
    )

    # ------------------------------------------------------------
    # Quasi-ELK timing
    # ------------------------------------------------------------
    quasi_elk_max_iters = 64

    run_and_cleanup(
        name="Quasi-ELK",
        fn=lambda: elk_alg(
            f=f,
            initial_state=initial_state,
            states_guess=make_states_guess(T, state_dim, device, dtype),
            drivers=drivers,
            sigmasq=1e8,
            process_noise=1.0,
            num_iters=quasi_elk_max_iters,
            tol=1e-12,
            quasi=True,
            damping=0.0,
            clip_value=1e8,
            return_trace=False,
        ),
        device=device,
        true_states_cpu=true_states_cpu,
    )

    del true_states_cpu
    clear_cuda_memory(device)


if __name__ == "__main__":
    main()
