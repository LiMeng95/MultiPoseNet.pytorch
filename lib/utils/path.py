import os
import shutil


def mkdir(path, rm_exist=False):
    if os.path.isdir(path):
        if not rm_exist:
            return
        shutil.rmtree(path)

    os.makedirs(path)
