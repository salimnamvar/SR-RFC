""" Neural Network Module Utility

"""

# region Imported Dependencies
from dataclasses import dataclass
# endregion Imported Dependencies


@dataclass
class Param:
    name: str
    kwargs: dict
