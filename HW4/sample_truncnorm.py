#! /usr/bin/env python

import time
import numpy as np
from scipy.stats import norm
from truncnorm import truncnorm


# initialize parameters for the truncated normal
n = 10000;
mu, sigma =  0, 1
lo, hi = -np.float32('inf'), -10
vals = np.zeros(n)
maxtries = 1000

start = time.time() # record start time
# Sample
for i in range(n):
    vals[i] = truncnorm(mu, sigma, lo, hi, maxtries)

end = time.time() # record end time
print 'CPU run time to sample: %f' % (end-start)

# verify outputs
print 'Mean of our drawn samples: %.6f' % np.mean(vals)
# compute theoretical truncated normal mean
rv = norm()
alpha, beta = (lo - mu)/sigma, (hi - mu)/sigma
theoretic_mean = mu + (rv.pdf(alpha) - rv.pdf(beta))/(rv.cdf(beta) - rv.cdf(alpha))*sigma
print 'Theoretical mean of truncated normal: %.6f' % theoretic_mean
