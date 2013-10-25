##
#
# Logistic regression
# 
# Y_{i} | \beta \sim \textrm{Bin}\left(n_{i},e^{x_{i}^{T}\beta}/(1+e^{x_{i}^{T}\beta})\right)
# \beta \sim N\left(\beta_{0},\Sigma_{0}\right)
#
##

import sys
import numpy as np

########################################################################################
## Handle batch job arguments:

nargs = len(sys.argv)
print 'Command line arguments: ' + str(sys.argv)

####
# sim_start ==> Lowest simulation number to be analyzed by this particular batch job
###

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

########################################################################################
########################################################################################



def bayes_logreg(m,y,X,beta_0,Sigma_0_inv,niter=10000,burnin=1000,print_every=1000,retune=100,verbose=False):
        
    def log_post_pdf(beta):
        return (np.dot(np.dot(X, beta), y) - 0.5*np.dot(beta-beta_0, np.dot(Sigma_0_inv, beta-beta_0))) - \
                np.dot(m, np.log(1+np.exp(np.dot(X, beta))))         

    p = X.shape[1]  # dimension of x's
    samples = np.zeros([niter+burnin, p])
    samples[0, :] = np.zeros(p)
                        
    for t in xrange(1, niter+   burnin):
            beta_new = np.random.multivariate_normal(samples[t-1, :], np.diag(np.ones(p)))
            alpha = log_post_pdf(beta_new) - log_post_pdf(samples[t-1, :])
            if np.log(np.random.rand()) < alpha:
                    samples[t, :] = beta_new
            else:
                    samples[t, :] = samples[t-1, :]

    return samples[burnin:, :]

#################################################
p = 2
beta_0 = np.zeros(2)
Sigma_0_inv = np.diag(np.ones(p))
# More stuff...
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
# print samples.shape
# exit()

# Extract posterior quantiles...
beta_percentiles = np.zeros([99, p])
for i in range(99):
    for r in range(p):
        beta_percentiles[i, r] = np.percentile(samples[:, r], i+1)

# Write results to a (99 x p) csv file...
res_dir = 'results/'
np.savetxt(res_dir + 'blr_res_' + str(sim_num) + '.csv', beta_percentiles, fmt='%.10f', delimiter=',')

# Go celebrate.
 
