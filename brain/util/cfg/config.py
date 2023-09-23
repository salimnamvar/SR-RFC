"""Configuration Parameters

    This file contains the struct classes of defining the configuration parameters.
"""


# region Imported Dependencies
from brain.util.text.type_parse import parser
# endregion Imported Dependencies


class CFG:
    def __init__(self):
        pass


class BrainConfig:
    __instance = None

    def __init__(self, a_cfg: str) -> None:

        if BrainConfig.__instance is not None:
            raise Exception('The `Config` class is allowed to have one instance.')
        else:
            self.__cfg_path: str = a_cfg

            # Parse Configuration file
            self.__parse_file()
            # Sub Configurations

            BrainConfig.__instance = self

    def __parse_file(self):
        with open(self.__cfg_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                key, value = line.split('=')
                keys = key.split('.')

                current_level = self

                for k in keys[:-1]:
                    if k not in current_level.__dict__:
                        obj = CFG()
                        current_level.__dict__[k] = obj
                    current_level = current_level.__dict__[k]
                current_level.__dict__[keys[-1]] = parser(value)

    @staticmethod
    def get_instance(a_cfg: str = None) -> 'BrainConfig':
        if BrainConfig.__instance is None:
            BrainConfig(a_cfg=a_cfg)
        return BrainConfig.__instance
