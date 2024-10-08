"""Base functionalities"""

import torch
import logging
from pathlib import Path
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from typing import TypeVar, Type
from omegaconf import OmegaConf


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


T = TypeVar("TBaseConfig", bound="BaseConfig")


class BaseConfig:
    """Base configuration class"""

    CONFIG_FILENAME = "config.yaml"

    def save_config(self, conf_path: Path):
        """Save config to model_dir"""
        conf = OmegaConf.to_yaml(self)
        conf_path.mkdir(parents=True, exist_ok=True)
        conf_path = conf_path / self.CONFIG_FILENAME
        OmegaConf.save(self, conf_path)
        return conf_path

    @classmethod
    def load_config(cls: Type[T], fpath: Path) -> T:
        """Load config from file path"""
        config = OmegaConf.load(fpath)
        conf = cls(**config)
        return conf


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
