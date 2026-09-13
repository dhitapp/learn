import torch.nn as nn

class PositionEmbedding(nn.Module):
    def __init__(self, 
                 seq_length:int,
                 embedding_size:int,
                 device):

        super().__init__()
        self.device = device
        self.seq_length = seq_length
        self.embedding_size = embedding_size
        self.position_embedding_table = nn.Embedding(seq_length, embedding_size)

    def forward(self, x):
        x = x.to(self.device)
        return self.position_embedding_table(x).to(self.device)