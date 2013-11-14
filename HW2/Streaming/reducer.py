#! /usr/bin/env python
# Sum all values of key-value pairs with the same given key
import sys

current_key, current_count = None, 0
BIN_SIZE = 0.1

for line in sys.stdin:
	# process line to get key and value
	line = line.strip()
	key, count = line.split('\t')
	count = int(count)

	# keep counting up if current key is the same as previous
	# otherwise, print out previous result and restart counting 
	if current_key == key:
		current_count += count
	else:
		if current_key:
			x_bin, y_bin = current_key.split(',')
			x_lo, x_hi = float(x_bin)*BIN_SIZE, (float(x_bin)+1)*BIN_SIZE
			y_lo, y_hi = float(y_bin)*BIN_SIZE, (float(y_bin)+1)*BIN_SIZE
			print '%.1f,%.1f,%.1f,%.1f,%d' % (x_lo, x_hi, y_lo, y_hi, current_count)
		current_key = key
		current_count = count

# print out result corresponding to last key
if current_key == key:
	x_bin, y_bin = current_key.split(',')
	x_lo, x_hi = float(x_bin)*BIN_SIZE, (float(x_bin)+1)*BIN_SIZE
	y_lo, y_hi = float(y_bin)*BIN_SIZE, (float(y_bin)+1)*BIN_SIZE
	print '%.1f,%.1f,%.1f,%.1f,%d' % (x_lo, x_hi, y_lo, y_hi, current_count)
