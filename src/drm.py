import pytorch_lightning as pl
import torch

# from loguru import logger


class UserTargetingModel(pl.LightningModule):
    def __init__(self, model, hyparams):
        super().__init__()

        self.model = model
        self.learning_rate = hyparams.learning_rate
        self.weight_decay = hyparams.weight_decay

    def forward(self, x, T):
        return self.model(x, T)

    def calculate_loss(self, batch, mode="train"):
        t_map = {1: 1, 0: -1}
        t = torch.tensor([t_map[t.item()] for t in batch["treatment"]]).float()
        g, c = batch["gain"], batch["cost"]

        p = self(batch["features"], batch["treatment"].to(torch.int64))
        p = torch.mul(p, t)

        agg_gain = torch.dot(g, p)
        agg_cost = torch.dot(c, p)
        loss = agg_cost / agg_gain
        # logger.debug(f"loss: {agg_cost} / {agg_gain} = {loss}")

        # Logging
        self.log_dict({f"{mode}_loss": loss}, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, "test")
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch["features"]
        T = batch["treatment"].to(torch.int64)

        p = self.model(x, T)

        return p

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

    def on_epoch_start(self):
        """Create a new progress bar for each epoch"""
        print("\n")


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
