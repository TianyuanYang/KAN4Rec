from torch import nn as nn

from models.kan4rec_modules.embedding import KAN4RecEmbedding
from models.kan4rec_modules.transformer import TransformerBlock
from utils import fix_random_seed_as


class KAN4Rec(nn.Module):
    def __init__(self, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        max_len = args.max_len
        num_items = args.num_items
        n_layers = args.num_blocks
        heads = args.num_heads
        vocab_size = num_items + 2
        hidden = args.hidden_units
        self.hidden = hidden
        dropout = args.dropout

        # embedding for KAN4Rec, sum of positional, segment, token embeddings
        self.embedding = KAN4RecEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass
