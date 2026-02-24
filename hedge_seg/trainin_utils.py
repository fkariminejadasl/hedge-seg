import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    # https://pytorch.org/docs/stable/notes/randomness.html
    # os.environ["PYTHONHASHSEED"] = str(seed) # PYTHONHASHSEED=42 python your_script.py
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multiple gpu
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # generator = torch.Generator().manual_seed(seed)  # for random_split
