"""
Train module

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from .base import TrainerBase, TRAINERS
from .default import Trainer
from .multi_dataset import MultiDatasetTrainer
from .wild_places import WildPlacesTrainer
# from .wildplaces_lc import WildPlacesLCTrainer

# __all__ = [
#     'TrainerBase',
#     'Trainer', 
#     'MultiDatasetTrainer',
#     'WildPlacesTrainer',
#     # 'WildPlacesLCTrainer',
#     'TRAINERS'
# ]
