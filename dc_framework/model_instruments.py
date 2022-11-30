import os
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

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


def _worker(rank, train_info, train_data, batch_size, world_size):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    device = f"cuda:{rank}"

    model = train_info["model"]
    criterion = train_info["criterion"]
    optimizer_kwargs = train_info["optimizer_kwargs"]

    model = torch.nn.parallel.DistributedDataParallel(model.to(device))

    loader = train_data.get_distributed_dataloader(batch_size)
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)

    loader_range = tqdm(loader) if rank == 0 else loader
    for feature, target in loader_range:
        optimizer.zero_grad()
        feature = feature.to(device)
        target = target.to(device)
        output = model(feature)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


class DCFramework:
    def __init__(
        self,
        model: torch.nn.Module, 
        criterion: torch.nn.Module, 
        lr=1e-3, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.lr = lr
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        self.criterion = criterion
        self.device = device
        self.gpus_count = torch.cuda.device_count()

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

    def train_single(self, train_data: Dict[str, np.array], batch_size: int = 1):
        train_data = Dataset(train_data)
        train_dataloader = train_data.get_dataloader(batch_size=batch_size)

        for batch in train_dataloader:
            output = self.forward(*batch)
            loss = output["loss"]
            loss.backward()
            self.optimizer.step()

    def train(self, train_data: Dict[str, np.array], batch_size: int = 1):
        if self.gpus_count > 1 and torch.distributed.is_available():
            self.train_parallel(train_data, batch_size)
        else:
            self.train_single(train_data, batch_size)

    def train_parallel(self, train_data, batch_size):
        train_data = Dataset(train_data)
        train_info = dict(
            model=self.model,
            criterion=self.criterion,
            optimizer_kwargs={"lr": self.lr}
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"

        torch.multiprocessing.spawn(
            _worker,
            nprocs=self.gpus_count,
            args=(train_info, train_data, batch_size, self.gpus_count)
        )

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
