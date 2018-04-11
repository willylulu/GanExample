import os
import sys
import numpy as np
from StackGan_v2 import StackGan

os.environ["CUDA_VISIBLE_DEVICES"]="1"
stackgan = StackGan(sys.argv[1], training=True)
stackgan.train()