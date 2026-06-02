from src.pararnn.config import DeerNewtonConfig, ParaRNNConfig, ParaRNNDeerConfig
from src.pararnn.base_cell import BaseParaRNNCell, BaseDeerRNNCell
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
    "DeerNewtonConfig",
    "ParaRNNConfig",
    "ParaRNNDeerConfig",
    "BaseParaRNNCell",
    "BaseDeerRNNCell",
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
