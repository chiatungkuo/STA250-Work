#! /usr/bin/env python2.7

import numpy as np
from truncnorm import truncnorm
#from pylab import *

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
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
    int success = 0;
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
                success = 1;
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
                success = 1;
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
				success = 1;
				break;
			}
			count++;        
    	}
    }
    if (!success) {
    	printf("No successful sample was drawn. Returning the mean.\\n");
    	vals[idx] = mu[idx];
    }

    return;
}

} // END extern "C"
""", include_dirs=['/usr/local/cuda/include/'], no_extern_c=1)

truncnorm_gpu = module.get_function('truncnorm_kernel');


def probit_mcmc(y, X, beta_0, sigma_0_inv, niter, burnin, print_every=1000, useGPU=False):
	n, p = X.shape
	beta = np.zeros(p) # all zeros to start with
	samples = np.zeros([niter, p])

	sigma_beta_inv = sigma_0_inv + np.dot(X.T, X)
	sigma_beta = np.linalg.solve(sigma_beta_inv, np.identity(p))
	maxtries = 100

	if useGPU:
		# block and grid sizes
		tpb = int(512)        # threads per block
		nb = int(1 + (n/tpb)) # number of blocks using 1D grid

	for t in range(niter + burnin):
        # sample conditional Z (but not stored)
		if useGPU: # use GPU verson truncnorm
			z = np.zeros(n, dtype=np.float32)
			mu_z, sigma_z = np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)
			lo, hi = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
			
			for i in range(n):
				if y[i] == 1:
					lo[i], hi[i] = np.float32(0), np.float32('inf')
				else:
					lo[i], hi[i] = -np.float32('inf'), np.float32(0)
				mu_z[i] = np.float32(np.dot(X[i, :], beta))

			truncnorm_gpu(drv.Out(z), np.int32(n), drv.In(mu_z), drv.In(sigma_z), drv.In(lo), drv.In(hi), \
				np.float32(maxtries), np.int32(np.random.randint(1e6)), block=(tpb, 1, 1), grid=(nb, 1))
		else: # use CPU verson truncnorm
			z = np.zeros(n)
			for i in range(n):
				if y[i] == 1:
					z[i] = truncnorm(np.dot(X[i, :], beta), 1, 0, np.float32('inf'), maxtries)
				else:
					z[i] = truncnorm(np.dot(X[i, :], beta), 1, -np.float32('inf'), 0, maxtries)

		sum_zx = np.dot(X.T, z)
		
        # sample conditional beta
		mu_beta = np.dot(sigma_beta, (np.dot(sigma_0_inv, beta_0) + sum_zx))
		beta = np.random.multivariate_normal(mu_beta, sigma_beta)
		if t >= burnin:
			samples[t-burnin, :] = beta
		if (t+1) % print_every == 0:
			print '%d iterations Finished.' % (t+1)

	return samples



# open data and extract y, m and X:
infile = open('data_04.txt', 'r')
lines = infile.readlines()
data = np.array(map(lambda x: x.rstrip('\n').split(), lines[1:]), dtype=np.float)
y = data[:, 0]
X = data[:, 1:]
infile.close()
n, p = X.shape

# initialize parameters
beta_0 = np.random.multivariate_normal(np.zeros(p), np.identity(p)) #np.zeros(p)
sigma_0_inv = np.identity(p)
niter, burnin = 2000, 500

# Gibbs sampling
beta_samples = probit_mcmc(y, X, beta_0, sigma_0_inv, niter, burnin, print_every=50, useGPU=True)

# print posterior means
for i in range(p):
	print np.mean(beta_samples[:, i])
