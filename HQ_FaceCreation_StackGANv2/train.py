import sys
import numpy as np
from StackGan_v2 import StackGan

stackgan = StackGan(sys.argv[1], training=True)
stackgan.train()