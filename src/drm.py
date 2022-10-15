import pytorch_lightning as pl
import torch
from loguru import logger


class UserTargetingModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, T):
        return self.model(x, T)

    def calculate_loss(self, g, c, p, T):
        t_map = {1: 1, 0: -1}
        t = torch.tensor([t_map[t.item()] for t in T]).float()
        p = torch.mul(p, t)

        agg_gain = torch.dot(g, p)
        agg_cost = torch.dot(c, p)

        logger.debug(f"agg_gain: {agg_gain}")
        logger.debug(f"agg_cost: {agg_cost}")


class DirectRankingModel(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.f1 = torch.nn.Linear(input_dim, hidden_dim)
        self.tanh = torch.nn.Tanh()
        self.f2 = torch.nn.Linear(hidden_dim, 1)
        self.scoring = Scoring()

    def forward(self, x, T):
        x = self.f1(x)
        x = self.tanh(x)
        x = self.f2(x)

        return self.scoring(x.flatten(), T)


class Scoring(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, T):
        sum_scores = torch.zeros(2).scatter_add(0, T, torch.exp(s))
        sum_map = {idx: score.item() for idx, score in enumerate(sum_scores)}
        output = torch.tensor([sum_map[t.item()] for t in T])

        return torch.div(torch.exp(s), output)
