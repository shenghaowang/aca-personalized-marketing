import torch

# from loguru import logger


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
