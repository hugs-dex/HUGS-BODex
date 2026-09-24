import os
import sys
import random
import logging
import traceback

import hydra
from omegaconf import DictConfig
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.task import *


@hydra.main(config_path="../cfg", config_name="base", version_base=None)
def main(cfg: DictConfig):
    eval(f"task_{cfg.task_name}")(cfg)


if __name__ == "__main__":
    main()
