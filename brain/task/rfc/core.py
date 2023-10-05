""" Experiment-1


"""


# region Imported Dependencies
from brain.task.base import BaseTask
from brain.util.cfg.config import BrainConfig
# endregion Imported Dependencies


class Task(BaseTask):
    def __init__(self):
        super().__init__()
        self.iteration: int = -1
        self.cfg: BrainConfig = BrainConfig.get_instance()

    def inference(self, *args, **kwargs):
        print('Task is ready')