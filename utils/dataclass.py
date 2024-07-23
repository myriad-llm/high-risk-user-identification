from dataclasses import dataclass
from torch import Tensor

@dataclass
class EmbeddingItem:
    embedding_name: str
    vocab_size: int
    embedding_dim: int
    x_col_index: Tensor
