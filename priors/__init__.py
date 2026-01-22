"""Priors package for synthetic tabular dataset generation.

Standalone implementation based on TabICL with improvements:
- Added min_classes parameter
- Fixed MulticlassAssigner to guarantee exact class count
"""

from .dataset import PriorDataset, SCMPrior, DummyPrior
from .dataloader import LocalPriorDataLoader, PriorDumpDataLoader, PriorDataLoader
from .utils import dump_prior_to_h5
from .reg2cls import Reg2Cls, MulticlassAssigner
from .prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP

__version__ = "0.1.0"

__all__ = [
    "PriorDataset",
    "SCMPrior", 
    "DummyPrior",
    "LocalPriorDataLoader",
    "PriorDumpDataLoader",
    "PriorDataLoader",
    "dump_prior_to_h5",
    "Reg2Cls",
    "MulticlassAssigner",
    "DEFAULT_FIXED_HP",
    "DEFAULT_SAMPLED_HP",
]
