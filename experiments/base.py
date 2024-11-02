"""Base functionalities"""

import logging
import os
import sys
from pathlib import Path
from typing import Type, TypeVar

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

dir_path = os.path.dirname(os.path.realpath(__file__))
SEED: int = 16  # Random seed
RESOURCES = Path(f"{dir_path}/../resources/")  # Path to resources
logging.basicConfig(level=logging.INFO)


class TBLogger(SummaryWriter):
    """Tensorboard Logger"""

    def __init__(self, log_dir: Path):
        super(TBLogger, self).__init__(log_dir)


def setup_logger() -> logging.Logger:
    """Set a console logger at Info level"""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        fmt = logging.Formatter("[%(name)s] [%(levelname)s] %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        logger.propagate = False  # Prevent duplicating msgs
    logger.info("Added console handler")
    return logger


logger = setup_logger()

logger.info("Resources path: %s", RESOURCES)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set device
logger.info("Torch device: %s", DEVICE)


def set_seeds(seed: int = SEED) -> None:
    """Set the seeds for all RNGs configured for the network"""
    torch.manual_seed(seed)  # Set CPU seed
    # Set GPU seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make torch algos deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Create a rng
    # rng = torch.Generator().manual_seed(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


BaseConfigT = TypeVar("BaseConfigT", bound="BaseConfig")


class BaseConfig:
    """Base configuration class"""

    CONFIG_FILENAME = "config.yaml"

    def save_config(self, conf_path: Path):
        """Save config to model_dir"""
        # conf = OmegaConf.to_yaml(self)
        conf_path.mkdir(parents=True, exist_ok=True)
        conf_path = conf_path / self.CONFIG_FILENAME
        OmegaConf.save(self, conf_path)
        return conf_path

    @classmethod
    def load_config(cls: Type[BaseConfigT], fpath: Path) -> BaseConfigT:
        """Load config from file path"""
        config = OmegaConf.load(fpath)
        conf = cls(**config)
        return conf


ModelT = TypeVar("ModelT", bound="BaseModel")


class BaseModel(torch.nn.Module):
    """Base model class with useful functions"""

    MODEL_FILENAME = "model.tar"
    CONFIG_FILENAME = "config.yaml"

    def save_model(self, model_dir: Path) -> tuple[Path, Path]:
        """Save model and config to model_path and return file locations"""

        model_dir.mkdir(parents=True, exist_ok=True)

        # save model
        model_path = model_dir / BaseModel.MODEL_FILENAME
        torch.save(self.state_dict(), model_path)
        logger.info("Model saved to %s ", model_path)

        # save params
        conf = OmegaConf.to_yaml(self.config)
        conf_path = model_dir / BaseModel.CONFIG_FILENAME
        with open(conf_path, "w") as f:
            f.write(conf)

        return model_path, conf_path

    @classmethod
    def load_model(cls: Type[ModelT], model_dir: Path) -> ModelT:
        """Load model from model_dir and return an instace of the model"""
        if not model_dir.is_dir():
            raise FileNotFoundError

        # load config
        with open(model_dir / BaseModel.CONFIG_FILENAME, "r") as f:
            config = OmegaConf.load(f)

        model = cls(config, tb_logger=None)
        # load state dict
        model_path = model_dir / BaseModel.MODEL_FILENAME
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)

        return model

    def count_parameters(self):
        """Count the number of parameters in a model"""
        num_params = count_parameters(self)
        logger.info("Num params: %s", num_params)


def copy_state_dict(from_model: torch.nn.Module, to_model: torch.nn.Module):
    """Load state dict from another model

    Build state dict using keys of to_model
    and values of from_model

    NOTE: This function should be used when the keys do not match,
    which could be because to_model is a custom implementation of a
    pretrained from_model.
    """
    to_state_dict = to_model.state_dict()
    from_state_dict = from_model.state_dict()

    state_dict = {
        to_key: from_value
        for (to_key, to_value), (from_key, from_value) in zip(
            to_state_dict.items(), from_state_dict.items()
        )
    }

    incompatible_keys = to_model.load_state_dict(state_dict)
    return to_model, incompatible_keys
