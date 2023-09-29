""" Project Path Utilities

"""


# region Imported Dependencies
import os
# endregion Imported Dependencies


def prj_path(a_path: str, a_marker: str):
    current_path = a_path if os.path.isdir(a_path) else os.path.dirname(a_path)
    if os.path.exists(os.path.join(current_path, a_marker)):
        path = current_path
    else:
        path = os.path.dirname(current_path)
        path = prj_path(a_path=path, a_marker=a_marker)
    return path
