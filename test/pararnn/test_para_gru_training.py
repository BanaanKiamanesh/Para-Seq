from src.pararnn.cells.para_gru import ParaGRUCell, ParaGRUConfig
from src.pararnn import DeerNewtonConfig
from torch import nn
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class TinyParaGRUClassifier(nn.Module):
    """Tiny sequence classifier used only for training tests.

    Architecture:

        x_{1:T} -> ParaGRU -> h_T -> Linear -> logits

    The target is deliberately simple:

        y = 1[ x_{T,0} > 0 ]

    This validates trainability without making the test slow or brittle.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        mode: str,
        num_iters: int,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()

        config = ParaGRUConfig(
            input_dim=input_dim,
            state_dim=state_dim,
            mode=mode,
            dtype=dtype,
            recurrent_init_scale=0.05,
            bias_init_value=0.0,
            deer=DeerNewtonConfig(
                num_iters=num_iters,
                tol=1e-8,
                strict_tol=False,
                stopping_criterion="update",
                initial_guess="f0",
                quasi=False,
                scan_backend="torch",
            ),
        )

        self.rnn = ParaGRUCell(config).to(dtype=dtype)
        self.readout = nn.Linear(state_dim, 2).to(dtype=dtype)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        hidden_sequence = self.rnn(x, mode=mode)
        final_hidden = hidden_sequence[:, -1, :]
        logits = self.readout(final_hidden)

        return logits


def make_last_value_classification_data(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    dtype: torch.dtype,
    seed: int,
):
    generator = torch.Generator().manual_seed(seed)

    x = 0.5 * torch.randn(
        batch_size,
        seq_len,
        input_dim,
        dtype=dtype,
        generator=generator,
    )

    y = (x[:, -1, 0] > 0.0).long()

    return x, y


def assert_all_parameter_grads_are_finite(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient for parameter {name}."
        assert torch.isfinite(param.grad).all(), (
            f"Non-finite gradient found in parameter {name}."
        )


def max_parameter_change(before, model: nn.Module) -> float:
    max_change = 0.0

    for name, param in model.named_parameters():
        change = torch.max(torch.abs(param.detach() - before[name])).item()
        max_change = max(max_change, change)

    return max_change


def train_classifier(
    mode: str,
    batch_size: int,
    seq_len: int,
    input_dim: int,
    state_dim: int,
    num_iters: int,
    steps: int,
    learning_rate: float,
    seed: int,
):
    torch.manual_seed(seed)

    dtype = torch.float64

    x, y = make_last_value_classification_data(
        batch_size=batch_size,
        seq_len=seq_len,
        input_dim=input_dim,
        dtype=dtype,
        seed=seed + 1000,
    )

    model = TinyParaGRUClassifier(
        input_dim=input_dim,
        state_dim=state_dim,
        mode=mode,
        num_iters=num_iters,
        dtype=dtype,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }

    with torch.no_grad():
        initial_logits = model(x, mode=mode)
        initial_loss = criterion(initial_logits, y).item()

    last_loss = None

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)

        logits = model(x, mode=mode)
        loss = criterion(logits, y)

        assert torch.isfinite(loss)

        loss.backward()

        assert_all_parameter_grads_are_finite(model)

        optimizer.step()

        last_loss = loss.item()

    with torch.no_grad():
        final_logits = model(x, mode=mode)
        final_loss = criterion(final_logits, y).item()
        final_accuracy = (final_logits.argmax(dim=-1)
                          == y).double().mean().item()

    parameter_change = max_parameter_change(before, model)

    return {
        "initial_loss": initial_loss,
        "last_training_loss": last_loss,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "parameter_change": parameter_change,
    }


def test_para_gru_training_loss_decreases_sequential_mode():
    result = train_classifier(
        mode="sequential",
        batch_size=8,
        seq_len=4,
        input_dim=2,
        state_dim=6,
        num_iters=8,
        steps=20,
        learning_rate=5e-2,
        seed=123,
    )

    assert result["parameter_change"] > 0.0
    assert result["final_loss"] < result["initial_loss"]
    assert result["final_loss"] < 0.40
    assert result["final_accuracy"] >= 0.875


def test_para_gru_training_loss_decreases_deer_mode():
    result = train_classifier(
        mode="deer",
        batch_size=4,
        seq_len=3,
        input_dim=2,
        state_dim=4,
        num_iters=8,
        steps=5,
        learning_rate=5e-2,
        seed=456,
    )

    assert result["parameter_change"] > 0.0
    assert result["final_loss"] < result["initial_loss"]
    assert result["final_accuracy"] >= 0.50
