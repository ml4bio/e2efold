from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import numpy as np


cmd_opt = argparse.ArgumentParser(description='Argparser for RNA structured prediction', allow_abbrev=False)
cmd_opt.add_argument('-gpu', type=int, default=1, help='-1: cpu; 0 - ?: specific gpu index')
cmd_opt.add_argument('-seed', type=int, default=19260817, help='seed')

cmd_opt.add_argument('--USE_CUDA_FLAG', type=int, default=0, help='USE GPU if = 1')
cmd_args = cmd_opt.parse_args()

print(cmd_args)
