"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

# Import all trainers from the new modular structure
from .train import (
    TrainerBase,
    Trainer,
    MultiDatasetTrainer,
    WildPlacesTrainer,
    # WildPlacesLCTrainer,
    TRAINERS
)

# Re-export for backward compatibility
__all__ = [
    'DefaultTrainer',
    'MultiDatasetTrainer', 
    'WildPlacesTrainer',
]
