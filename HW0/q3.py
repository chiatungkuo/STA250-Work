#! /usr/bin/env python

import sys

infile = open(sys.argv[1], 'r')
counter = 0
while True:
	ch = infile.read(1)
	if ch == '':
		break

	counter += 1
	outFileName = "out_" + str(counter) + ".txt"
	outfile = open(outFileName, 'w')
	outfile.write(ch)
	outfile.close()
