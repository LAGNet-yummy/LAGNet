import torch
import random
import numpy as np
from torch.backends import cudnn


def setSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark=False
    cudnn.deterministic=True
    random.seed(seed)


if __name__=="__main__":
    setSeed(0)
    print(torch.randn((1)))
    print(random.random())
    print(np.random.randn(1))
