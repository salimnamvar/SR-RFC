""" Stanford Ribonanza RNA Folding Challenge [SR-RFC]

    This file is a starting point for running the STAR application.
"""


# region Imported Dependencies------------------------------------------------------------------------------------------
import os
from pathlib import Path
import argparse
from brain.core import Sys
# endregion Imported Dependencies


if __name__ == '__main__':
    # Input Parameters
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='SR-RFC')
    parser.add_argument('--cfg', '-c', type=str, help='The configuration file path',
                        default=os.path.join(project_root, 'cfg', 'cfg.properties'))
    args = parser.parse_args()

    # Application Initialization
    app = Sys(a_cfg=args.cfg)

    # Application Run
    app.inference()


