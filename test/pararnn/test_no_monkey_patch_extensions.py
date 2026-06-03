from pathlib import Path

import torch

from src.pararnn import ParaGRU, ParaLSTM, ParaRNN


def test_old_extension_blocks_were_removed():
    forbidden_markers = [
        "BaseParaRNNCell ELK layer API extension",
        "ParaGRU ELK layer API extension",
        "ParaRNN ELK layer API extension",
        "ParaLSTM block-2 custom adjoint extension",
        "ParaLSTM ELK layer API extension",
    ]

    files = [
        Path("src/pararnn/base_cell.py"),
        Path("src/pararnn/cells/para_gru.py"),
        Path("src/pararnn/cells/para_rnn.py"),
        Path("src/pararnn/cells/para_lstm.py"),
    ]

    for file in files:
        text = file.read_text()
        for marker in forbidden_markers:
            assert marker not in text, f"Old extension marker still exists in {file}: {marker}"


def test_gru_elk_config_helper_exists_and_runs():
    torch.manual_seed(5101)

    model = ParaGRU(
        input_size=3,
        hidden_size=4,
        batch_first=True,
        mode="elk",
        deer_config=__import__(
            "src.pararnn.cells.para_gru",
            fromlist=["make_paragru_elk_config"],
        ).make_paragru_elk_config(num_iters=3),
        dtype=torch.float64,
    )

    x = torch.randn(2, 5, 3, dtype=torch.float64)
    y, h = model(x)

    assert y.shape == (2, 5, 4)
    assert h.shape == (1, 2, 4)
    assert model.last_deer_infos[-1]["solver"] == "elk"


def test_rnn_elk_config_helper_exists_and_runs():
    torch.manual_seed(5102)

    from src.pararnn.cells.para_rnn import make_pararnn_elk_config

    model = ParaRNN(
        input_size=3,
        hidden_size=4,
        batch_first=True,
        mode="elk",
        deer_config=make_pararnn_elk_config(backend="quasi_elk", num_iters=3),
        dtype=torch.float64,
    )

    x = torch.randn(2, 5, 3, dtype=torch.float64)
    y, h = model(x)

    assert y.shape == (2, 5, 4)
    assert h.shape == (1, 2, 4)
    assert model.last_deer_infos[-1]["solver"] == "elk"


def test_lstm_block_adjoint_and_elk_helpers_still_run():
    torch.manual_seed(5103)

    from src.pararnn.cells.para_lstm import make_paralstm_deer_config, make_paralstm_elk_config

    x = torch.randn(2, 5, 3, dtype=torch.float64)

    deer = ParaLSTM(
        input_size=3,
        hidden_size=4,
        batch_first=True,
        mode="deer",
        deer_config=make_paralstm_deer_config(
            backend="adjoint",
            num_iters=3,
        ),
        dtype=torch.float64,
        recurrent_init_scale=0.025,
        peephole_init_scale=0.025,
    )

    y_deer, (h_deer, c_deer) = deer(x)

    assert y_deer.shape == (2, 5, 4)
    assert h_deer.shape == (1, 2, 4)
    assert c_deer.shape == (1, 2, 4)
    assert deer.last_deer_infos[-1]["backward_backend"] == "adjoint"

    elk = ParaLSTM(
        input_size=3,
        hidden_size=4,
        batch_first=True,
        mode="elk",
        deer_config=make_paralstm_elk_config(num_iters=3),
        dtype=torch.float64,
        recurrent_init_scale=0.025,
        peephole_init_scale=0.025,
    )

    y_elk, (h_elk, c_elk) = elk(x)

    assert y_elk.shape == (2, 5, 4)
    assert h_elk.shape == (1, 2, 4)
    assert c_elk.shape == (1, 2, 4)
    assert elk.last_deer_infos[-1]["solver"] == "elk"
