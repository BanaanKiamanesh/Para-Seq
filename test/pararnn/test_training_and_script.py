import csv
import json

import torch
from torch import nn

from src.pararnn import ParaGRU, make_paragru_deer_config


class TinySequenceClassifier(nn.Module):
    def __init__(self, *, backend="adjoint", dtype=torch.float64):
        super().__init__()
        self.rnn = ParaGRU(
            input_size=3,
            hidden_size=5,
            batch_first=True,
            mode="deer",
            backend=backend,
            scan_backend="torch",
            num_iters=8 if dtype == torch.float64 else 4,
            tol=1e-10 if dtype == torch.float64 else 1e-4,
            strict_tol=(dtype == torch.float64),
            dtype=dtype,
            recurrent_init_scale=0.08,
        )
        self.head = nn.Linear(5, 2).to(dtype=dtype)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.head(h_n[-1])


def test_adjoint_gru_training_step_and_config_factory():
    cfg = make_paragru_deer_config(
        backend="adjoint",
        scan_backend="accel_scan",
        num_iters=4,
        tol=1e-4,
    )
    assert (cfg.quasi, cfg.scan_backend, cfg.backward_backend) == (
        True,
        "accel_scan",
        "adjoint",
    )

    torch.manual_seed(0)
    x = 0.5 * torch.randn(8, 7, 3, dtype=torch.float64)
    y = (x[:, -1, 0] > 0.0).long()
    model = TinySequenceClassifier(backend="adjoint", dtype=torch.float64)

    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.rnn.last_deer_infos[0]["backward_backend"] == "adjoint"
    assert all(param.grad is not None for param in model.parameters())


def test_synthetic_training_export_smoke(tmp_path):
    torch.manual_seed(1)

    output_dir = tmp_path / "run"
    output_dir.mkdir()

    x = 0.5 * torch.randn(8, 6, 3, dtype=torch.float32)
    y = (x[:, -1, 0] > 0.0).long()
    model = TinySequenceClassifier(backend="adjoint", dtype=torch.float32)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    record = {
        "backend": "adjoint_torch",
        "epoch": 1,
        "train_loss": float(loss.detach().item()),
        "train_accuracy": float((logits.argmax(dim=-1) == y).float().mean()),
    }
    csv_path = output_dir / "smoke_history.csv"
    json_path = output_dir / "smoke_history.json"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)

    json_path.write_text(json.dumps([record]), encoding="utf-8")

    assert torch.isfinite(loss)
    assert csv_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["epoch"] == 1


def test_loss_decreases_on_tiny_problem():
    torch.manual_seed(2)
    x = 0.5 * torch.randn(16, 8, 3, dtype=torch.float64)
    y = (x[:, -1, 0] + x[:, :, 1].mean(dim=1) > 0.0).long()

    model = TinySequenceClassifier(backend="adjoint", dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        initial_loss = criterion(model(x), y).item()

    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = criterion(model(x), y).item()

    assert final_loss < initial_loss
