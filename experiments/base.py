"""Base functionalities"""

import torch
import logging
from pathlib import Path
import os
import sys
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
