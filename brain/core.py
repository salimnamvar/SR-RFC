""" AIBrain Core

    This file defines an application's core to handle the tasks.
"""


# region Imported Dependencies
import importlib
from brain.task.base import BaseTask
from brain.util.cfg.config import BrainConfig
from brain.util.log.setup import SetupLogger
# endregion Imported Dependencies


class Sys:
    def __init__(self, a_cfg: str):
        self.cfg: BrainConfig = BrainConfig.get_instance(a_cfg=a_cfg)
        self.task: BaseTask = None
        self.__create_task()
        SetupLogger(a_cfg=self.cfg)

    def __create_task(self):
        module_name = f"brain.task.{self.cfg.tsk.name.lower()}.core"
        module = importlib.import_module(module_name)
        task_class = getattr(module, 'Task')
        self.task: BaseTask = task_class()

    def inference(self):
        try:
            task_method = getattr(self.task, self.cfg.tsk.method.lower())
        except (AttributeError, ModuleNotFoundError):
            raise ValueError('Invalid `tsk.method` is entered.')
        task_method()
