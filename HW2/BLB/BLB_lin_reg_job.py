import sys
import csv
import numpy as np
import wls
import read_some_lines as rsl

print "=============================="
print "Python version:"
print str(sys.version)
print "Numpy version:"
print str(np.version.version)
print "=============================="

mini = False
verbose = False

datadir = "/home/pdbaines/data/"
outpath = "output/"

# mini or full?
if mini:
	rootfilename = "blb_lin_reg_mini"
else:
	rootfilename = "blb_lin_reg_data"

########################################################################################
## Handle batch job arguments:

nargs = len(sys.argv)
print 'Command line arguments: ' + str(sys.argv)

####
# sim_start ==> Lowest simulation number to be analyzed by this particular batch job
###

#######################
sim_start = 1000
length_datasets = 250
#######################

# Note: this only sets the random seed for numpy, so if you intend
# on using other modules to generate random numbers, then be sure
# to set the appropriate RNG here

if (nargs==1):
	sim_num = sim_start + 1
	sim_seed = 1330931
else:
	# Decide on the job number, usually start at 1000:
	sim_num = sim_start + int(sys.argv[nargs-1])
	# Set a different random seed for every job number!!!
	sim_seed = 762*sim_num + 1330931

# Set the seed:
np.random.seed(sim_seed)

# Bootstrap datasets numbered 1001-1250

########################################################################################
########################################################################################

# Find r and s indices:
r_index = (sim_num - sim_start - 1) % 50 + 1
s_index = int((sim_num - sim_start - 1) / 50) + 1 

print "========================="
print "sim_num = " + str(sim_num)
print "s_index = " + str(s_index)
print "r_index = " + str(r_index)
print "========================="

# define parameters as specified in the question
file_extension = '.txt'
gamma = 0.7
n, nc = 1000000, 1001
b = np.ceil(n**gamma)

# draw indices for a bootstrap dataset (same seed for same s_index)
np.random.seed(s_index*1330931)
indices = np.random.choice(range(n), size=b, replace=False)
indices.sort()
subset = rsl.read_some_lines_csv(datadir + rootfilename + file_extension, indices, len(indices), nc, n)

# reseed random and draw the number of occurrences for each data point
np.random.seed()
weights = np.random.multinomial(n, 1/b*np.ones(b))

# fit weighted linear regression
X, y = subset[:,:-1], subset[:, -1]
beta_hat = wls.wls(y, X, weights)

# write result to files
outfile = outpath + 'coef_%02d_%02d.txt' % (s_index, r_index)
np.savetxt(outfile, beta_hat, fmt='%.10f', delimiter=',')
