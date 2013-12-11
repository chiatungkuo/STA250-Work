#! /usr/bin/env python

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from scipy.stats import norm
from pycuda.compiler import SourceModule

module = SourceModule("""
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>
#include <math_constants.h>

extern "C"
{

__global__ void 
truncnorm_kernel(float *vals, int n, 
                  float *mu, float *sigma, 
                  float *lo, float *hi,
                  int maxtries, int rng_param)
{
    // Usual block/thread indexing...
    int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    int idx = myblock * blocksize + subthread;
	if (idx >= n) 
		return;

    // Setup the RNG:
    curandState rng;
    curand_init((unsigned long long) idx*rng_param, 0, 0, &rng);
    
    // Sample:
    // handle corner cases using approach by Robert 2009
    if (hi[idx] == CUDART_INF_F && isfinite(lo[idx])) {
    	//printf("left truncation\\n");
        int count = 0;
        while (count < maxtries) {
            float mu_bar = (lo[idx] - mu[idx])/sigma[idx];
            float alpha = (mu_bar + sqrtf(powf(mu_bar, 2.0F)+4.0F))/2.0F;
            float z = mu_bar -logf(curand_uniform(&rng))/alpha;
            float phi = expf(-powf(alpha-z, 2.0F)/2.0F);
            if (mu_bar >= alpha)
                phi *= expf(-powf(mu_bar-alpha, 2.0F)/2.0F);
            if (curand_uniform(&rng) < phi) {
                vals[idx] = mu[idx] + sigma[idx]*z;
                break;
            }
            count++;
        }
    }
    else if (isfinite(hi[idx]) && lo[idx] == -CUDART_INF_F) {
    	//printf("right truncation\\n");
        int count = 0;
        while (count < maxtries) {
            float mu_bar = (-hi[idx] + mu[idx])/sigma[idx];
            float alpha = (mu_bar + sqrtf(powf(mu_bar, 2.0F)+4.0F))/2.0F;
            float z = mu_bar -logf(curand_uniform(&rng))/alpha;
            float phi = expf(-powf(alpha-z, 2.0F)/2.0F);
            if (mu_bar >= alpha)
                phi *= expf(-powf(mu_bar-alpha, 2.0F)/2.0F);
            if (curand_uniform(&rng) < phi) {
                vals[idx] = -(-mu[idx] + sigma[idx]*z);
                break;
            }
            count++;
        }
    }
    else {
    	// printf("%f\\t%f\\n", lo[idx], hi[idx]);
    	int count = 0;
    	while (count < maxtries) {
			float x = mu[idx] + sigma[idx]*curand_normal(&rng);
			if (x >= lo[idx] && x <= hi[idx]) {
				vals[idx] = x;
				break;
			}
			count++;        
    	}
    }

    return;
}

} // END extern "C"
""", include_dirs=['/usr/local/cuda/include/'], no_extern_c=1)

truncnorm = module.get_function('truncnorm_kernel');

# initialize parameters for the truncated normal
n = np.int32(10000)
mu, sigma = np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)
lo, hi = -np.float32('inf')*np.ones(n, dtype=np.float32), -10*np.ones(n, dtype=np.float32)
vals = np.zeros_like(mu)
maxtries = np.int32(1000)

# choose block and grid sizes
tpb = int(512)        # threads per block
nb = int(1 + (n/tpb)) # number of blocks using 1D grid

# Sample
start, end = drv.Event(), drv.Event()
start.record() # record start time
truncnorm(drv.Out(vals), n, drv.In(mu), drv.In(sigma), drv.In(lo), drv.In(hi), maxtries, \
	np.int32(np.random.randint(1e6)), block=(tpb, 1, 1), grid=(nb, 1))
end.record() # record end time
end.synchronize()
gpu_secs = start.time_till(end)*1e-3
print 'GPU run time to sample: %f' % gpu_secs

# verify outputs
print 'Mean of our drawn samples: %.6f' % np.mean(vals)
rv = norm()
alpha, beta = (lo[0] - mu[0])/sigma[0], (hi[0] - mu[0])/sigma[0]
theoretic_mean = mu[0] + (rv.pdf(alpha) - rv.pdf(beta))/(rv.cdf(beta) - rv.cdf(alpha))*sigma[0]
print 'Theoretical mean of truncated normal: %.6f' % theoretic_mean
