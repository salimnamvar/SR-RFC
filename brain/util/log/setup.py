""" Logging
"""


# region Imported Dependencies
import logging

from brain.util.cfg.config import BrainConfig


# endregion Imported Dependencies


class SetupLogger:
    def __init__(self, a_filename: str):
        self.cfg: BrainConfig = BrainConfig.get_instance()
        self.filename: str = a_filename
        self.name: str = self.cfg.log.name
        self.level: str = self.cfg.log.level
        self.format: str = self.cfg.log.format
        self.levels: dict = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
            "FATAL": logging.FATAL
        }
        self.logger: logging.Logger = None
        self.__setup()

    def __setup(self) -> None:
        try:
            self.logger = logging.getLogger(self.name)
            level = self.levels.get(self.level.upper(), logging.INFO)
            self.logger.setLevel(level)
            file_handler = logging.FileHandler(self.filename)
            file_handler.setLevel(level)
            formatter = logging.Formatter(self.format)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f'Error setting up logger: {e}')