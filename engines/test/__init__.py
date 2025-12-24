"""
Train module

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from .base import TesterBase, TESTERS
from .clstest import ClsTester
from .clsvotetest import ClsVotingTester
from .semsegtest import SemSegTester
from .dino_semsegtest import DINOSemSegTester
from .parsegtest import PartSegTester
from .semsegtest_assemable import SemSegTester_Assemble
from .wild_places import WildPlacesTester