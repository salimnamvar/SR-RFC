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
        NotImplementedError('Subclasses should implement this method.')

    def train(self, *args, **kwargs):
        NotImplementedError('Subclasses should implement this method.')

    def test(self, *args, **kwargs):
        NotImplementedError('Subclasses should implement this method.')
