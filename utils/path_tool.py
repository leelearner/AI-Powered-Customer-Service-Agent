"""
Provide the absolute path for this project
"""

import os


def get_project_root() -> str:
    """
    Get the root dir of this project

    Returns:
        str: path of root dir
    """
    # The absolute path of the current file.
    current_file = os.path.abspath(__file__)
    # Get the root dir, first get the path of the dir where the file exists
    current_dir = os.path.dirname(current_file)
    # Second, get the root dir
    project_root = os.path.dirname(current_dir)

    return project_root


def get_abs_path(relative_path: str) -> str:
    """Get the absolute path from the relative path

    Args:
        relative_path (str): The relative path of the target file

    Returns:
        str: The absolute path of the file corresponding to the relative_path
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)


if __name__ == "__main__":
    print(get_abs_path("config/config.txt"))
