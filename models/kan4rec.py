from .base import BaseModel
from .kan4rec_modules.kan4rec import KAN4Rec

import torch.nn as nn


class KAN4RecModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.kan4rec = KAN4Rec(args)
        self.out = nn.Linear(self.kan4rec.hidden, args.num_items + 1)

    @classmethod
    def code(cls):
        return 'kan4rec'

    def forward(self, x):
        x = self.kan4rec(x)
        return self.out(x)
