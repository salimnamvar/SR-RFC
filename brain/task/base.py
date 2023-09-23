""" Base Task

    This file defines an abstract base brain task.
"""


# region Imported Dependencies
from abc import ABC
from brain.util.cfg.config import BrainConfig
# endregion Imported Dependencies


class BaseTask(ABC):
    def __init__(self, *args, **kwargs):
        self.cfg: BrainConfig = BrainConfig.get_instance()

    def inference(self, *args, **kwargs):
        NotImplemented

    def train(self, *args, **kwargs):
        NotImplemented

    def test(self, *args, **kwargs):
        NotImplemented
