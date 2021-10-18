import logging
import math
import random
import numpy as np
import time
import torch as t
import torch.nn as nn
from torch import optim
from torch.nn import utils
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)



state = t.tensor([1., 2])

step_mat = t.tensor([[1., 3],
                     [0., 1.]])
state = t.matmul(step_mat, state)
print(state[0])
