#! /usr/bin/env python
# Keys are x_bin,y_bin and value is 1 for each data point (x, y)
import sys
import numpy as np

BIN_SIZE = 0.1

for line in sys.stdin:
	line = line.strip()
	x, y = line.split()
	x_bin, y_bin = np.floor([float(x)/BIN_SIZE, float(y)/BIN_SIZE])
	print '%s\t%s' % (str(x_bin) + ',' + str(y_bin), '1')
