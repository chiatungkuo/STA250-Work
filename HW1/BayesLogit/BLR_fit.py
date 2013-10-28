import sys
import numpy as np
from bayes_logreg import bayes_logreg
from pylab import *

########################################################################################
## Handle batch job arguments:

nargs = len(sys.argv)
#print 'Command line arguments: ' + str(sys.argv)

#######################
sim_start = 1000
length_datasets = 200
#######################

# Note: this only sets the random seed for numpy, so if you intend
# on using other modules to generate random numbers, then be sure
# to set the appropriate RNG here

if (nargs<=1):
	sim_num = sim_start + 1
	np.random.seed(1330931)
else:
	# Decide on the job number, usually start at 1000:
	sim_num = sim_start + int(sys.argv[1])
	# Set a different random seed for every job number!!!
	np.random.seed(762*sim_num + 1330931)

# Simulation datasets numbered 1001-1200

#################################################
p = 2
beta_0 = np.zeros(2)
Sigma_0_inv = np.diag(np.ones(p))
#################################################

# Read data corresponding to appropriate sim_num:
data_dir = 'data/'
infile = open(data_dir + 'blr_data_' + str(sim_num) + '.csv', 'r')

# Extract y, m and X:
lines = infile.readlines()
data = np.array(map(lambda x: x.rstrip('\n').split(','), lines[1:]), dtype='float')
y = data[:, 0]
m = data[:, 1]
X = data[:, [2, 3]]
infile.close()

# Fit the Bayesian model:
samples = bayes_logreg(m, y, X, beta_0, Sigma_0_inv, niter=100000, burnin=5000)
# plot(samples)

# Extract posterior quantiles...
beta_percentiles = np.zeros([99, p])
for i in range(99):
    for r in range(p):
        beta_percentiles[i, r] = np.percentile(samples[:, r], i+1)

# Write results to a (99 x p) csv file...
res_dir = 'results/'
np.savetxt(res_dir + 'blr_res_' + str(sim_num) + '.csv', beta_percentiles, \
    fmt='%.10f', delimiter=',')

