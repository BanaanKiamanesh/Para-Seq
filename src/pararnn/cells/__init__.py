from src.pararnn.cells.tanh_deer_cell import TanhDeerRNNCell
from src.pararnn.cells.para_rnn import (
    ParaRNN,
    ParaRNNCell,
    ParaRNNBackend,
    ParaRNNNonlinearity,
    make_pararnn_deer_config,
)
from src.pararnn.cells.para_gru import ParaGRU, ParaGRUCell, ParaGRUConfig
try:
    from src.pararnn.cells.para_gru import ParaGRUBackend, make_paragru_deer_config
except ImportError:  # compatibility with older local snapshots
    ParaGRUBackend = str  # type: ignore
    make_paragru_deer_config = None  # type: ignore
try:
    from src.pararnn.cells.para_lstm import (
        ParaLSTM,
        ParaLSTMCell,
        ParaLSTMConfig,
        ParaLSTMBackend,
        make_paralstm_deer_config,
    )
except ImportError:  # compatibility with snapshots before ParaLSTM exists
    ParaLSTM = None  # type: ignore
    ParaLSTMCell = None  # type: ignore
    ParaLSTMConfig = None  # type: ignore
    ParaLSTMBackend = str  # type: ignore
    make_paralstm_deer_config = None  # type: ignore

__all__ = [
    "TanhDeerRNNCell",
    "ParaRNN",
    "ParaRNNCell",
    "ParaRNNBackend",
    "ParaRNNNonlinearity",
    "make_pararnn_deer_config",
    "ParaGRU",
    "ParaGRUCell",
    "ParaGRUConfig",
    "ParaGRUBackend",
    "make_paragru_deer_config",
    "ParaLSTM",
    "ParaLSTMCell",
    "ParaLSTMConfig",
    "ParaLSTMBackend",
    "make_paralstm_deer_config",
]

# === Native ELK exports ===
try:
    from src.pararnn.cells.para_gru import make_paragru_elk_config
    from src.pararnn.cells.para_rnn import make_pararnn_elk_config
    from src.pararnn.cells.para_lstm import make_paralstm_elk_config
except ImportError:
    pass
else:
    for _name in [
        "make_paragru_elk_config",
        "make_pararnn_elk_config",
        "make_paralstm_elk_config",
    ]:
        if _name not in __all__:
            __all__.append(_name)
# === End native ELK exports ===

# === Part-5 fix: exports ===
try:
    from src.pararnn.cells.para_gru import make_paragru_elk_config
    from src.pararnn.cells.para_rnn import make_pararnn_elk_config
    from src.pararnn.cells.para_lstm import make_paralstm_elk_config
except ImportError:
    pass
else:
    for _name in [
        "make_paragru_elk_config",
        "make_pararnn_elk_config",
        "make_paralstm_elk_config",
    ]:
        if _name not in __all__:
            __all__.append(_name)
# === End Part-5 fix: exports ===


# === Unified algorithm mode exports ===
try:
    from src.pararnn.cells.para_gru import (
        make_paragru_jacobi_config,
        make_paragru_picard_config,
    )
    from src.pararnn.cells.para_rnn import (
        make_pararnn_jacobi_config,
        make_pararnn_picard_config,
    )
    from src.pararnn.cells.para_lstm import (
        make_paralstm_jacobi_config,
        make_paralstm_picard_config,
    )
except ImportError:
    pass
else:
    for _name in [
        "make_paragru_jacobi_config",
        "make_paragru_picard_config",
        "make_pararnn_jacobi_config",
        "make_pararnn_picard_config",
        "make_paralstm_jacobi_config",
        "make_paralstm_picard_config",
    ]:
        if _name not in __all__:
            __all__.append(_name)
# === End unified algorithm mode exports ===
