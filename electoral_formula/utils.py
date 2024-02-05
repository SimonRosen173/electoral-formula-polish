import os
import shutil
import glob
from typing import List


def create_or_clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    os.mkdir(folder_path)


def clear_folder(folder_path):
    files = glob.glob(f'{folder_path}/*')
    for f in files:
        os.remove(f)


def join_paths(paths: List[str]):
    return os.path.normpath(os.path.join(*paths))
