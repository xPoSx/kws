import torch
import dataclasses
from typing import Tuple


@dataclasses.dataclass
class DistillationTaskConfig:
    keyword: str = 'sheila'  # We will use 1 key word -- 'sheila'
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50
    n_mels: int = 40
    cnn_out_channels: int = 8
    kernel_size: Tuple[int, int] = (12, 12)
    stride: Tuple[int, int] = (9, 9)
    hidden_size: int = 16
    gru_num_layers: int = 2
    bidirectional: bool = False
    num_classes: int = 2
    sample_rate: int = 16000
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    temperature: int = 7
    alpha = 0.3
