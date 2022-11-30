import logging
import numpy as np

import torch

from pathlib import Path
from typing import Dict

from dc_framework.data_preparation import Dataset

logger = logging.getLogger("__name__")


def init(model: torch.nn.Module, criterion: torch.nn.Module, *args, **kwargs):
    return DCFramework(model, criterion, *args, **kwargs)


def load_from_checkpoint(checkpoint_path: Path, *args, **kwargs):
    state = torch.load(checkpoint_path)
    trainer = DCFramework(*args, **kwargs)
    trainer.model.load_state_dict(state["model"])
    trainer.optimizer.load_state_dict(state["optimizer"])

    return trainer


class DCFramework:
    def __init__(
        self,
        model: torch.nn.Module, 
        criterion: torch.nn.Module, 
        lr=1e-3, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = criterion
        self.device = device

    def forward(self, feature, target, train=True):
        feature = feature.to(self.device)
        target = target.to(self.device)
        try:
            output = self.model(feature)
        except:
            logger.warning(f"feature: {feature}")
            raise
        if train:
            try:
                loss = self.criterion(output, target)
            except:
                logger.warning(f"output: {output}")
                logger.warning(f"target: {target}")
                raise
        result = {
            "output": output,
        }
        if train:
            result["loss"] = loss
        return result

    def train(self, train_data: Dict[str, np.array], batch_size: int = 1):
        train_data = Dataset(train_data)
        train_dataloader = train_data.get_dataloader(batch_size=batch_size)
        
        for batch in train_dataloader:
            output = self.forward(*batch)
            loss = output["loss"]
            loss.backward()
            self.optimizer.step()

    def validate(self, val_data: Dict[str, np.array], batch_size: int = 1, metric: str = "mse"):
        target = val_data["target"]
        prediction = self.test(val_data, batch_size)
        return self.calculate_metrics(prediction, target, metric)

    def calculate_metrics(self, prediction: np.array, target: np.array, metric: str = "mse"):
        if metric == "mse":
            return np.mean((prediction - target) ** 2)
        elif metric == "accuracy":
            return np.mean(prediction == target)
        else:
            raise ValueError("Unsupported metric:", metric)

    def test(self, test_data: Dict[str, np.array], batch_size: int = 1):
        test_data = Dataset(test_data)
        test_dataloader = test_data.get_dataloader(batch_size=batch_size)
        
        predictions = []
        for batch in test_dataloader:
            output = self.forward(*batch, train=False)["output"]
            predictions.append(output)
        return torch.cat(predictions).detach().cpu().squeeze().numpy()

    def save(self, path: Path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)
