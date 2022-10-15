import numpy as np
import torch
from loguru import logger


class Scoring(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, T):
        sum_scores = torch.zeros(2).scatter_add(0, T, torch.exp(s))
        # output = T.apply_(lambda x: sum_scores.numpy()[x])
        # output = torch.tensor(np.array([sum_scores[x] for x in T.numpy()]))

        func = np.vectorize(lambda x: sum_scores[x])
        output = torch.tensor(func(T.numpy()))
        return torch.div(torch.exp(s), output)


# class DirectRankingModel()


if __name__ == "__main__":
    scoring = Scoring()
    s = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    T = torch.tensor([1, 1, 0, 0, 1])
    # T = torch.tensor([1,1,0,0,1], dtype=torch.int64)
    res = scoring(s, T)

    logger.info(res)
