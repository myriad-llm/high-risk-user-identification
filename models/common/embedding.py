import torch.nn as nn
import torch
import pickle as pkl

class CallRecordsEmbeddings(nn.Module):

    def __init__(
        self,
        embedding_items_path,
        padding_idx: int = 0,
    ):

        # from utils.dataclass import EmbeddingItem
        super().__init__()
        if embedding_items_path == "fake":
            # HACK: for fake init
            print("fake init")
            return

        with open(embedding_items_path, 'rb') as f:
            embedding_items = pkl.load(f)

        self.embedding_items = embedding_items
        self.embeddings = nn.ModuleDict(
            {
                embedding_item.embedding_name: nn.Embedding(
                    embedding_item.vocab_size,
                    embedding_item.embedding_dim,
                    padding_idx=padding_idx,
                )
                for embedding_item in embedding_items
            }
        )

    def forward(self, x):
        b, seq, f_dim = x.shape
        embedding_list = []
        for embedding_item in self.embedding_items:
            before_embedding = x[:, :, embedding_item.x_col_index].long() # b * seq_len * len(embedding_item.x_col_index)
            embedding = self.embeddings[embedding_item.embedding_name](before_embedding) # b * seq_len * len(embedding_item.x_col_index) * embedding_dim
            embedding = embedding.reshape(b, seq, -1) # b * seq_len * (len(embedding_item.x_col_index) * embedding_dim)
            embedding_list.append(embedding)
        embeddings = torch.cat(embedding_list, dim=-1)

        del_idx = [embedding_item.x_col_index for embedding_item in self.embedding_items]
        del_idx = torch.cat(del_idx, dim=-1)
        mask = torch.ones(f_dim, dtype=torch.bool)
        mask[del_idx] = False
        embedding_x = torch.cat([x[:, :, mask], embeddings], dim=-1)
        if embedding_x.max() > 1 or embedding_x.min() < -1:
            print(embedding_x.max(), embedding_x.min())
        # 对 embedding 结果进行进一步处理
        
        

        return embedding_x