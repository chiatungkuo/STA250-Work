import numpy as np

""" Perform Metropolis within Gibbs sampling on the Bayesian logistic regression model
        y = binom(m, logit(x' * beta)) where logit is the logistic function and
        beta = multivariate_normal(beta_0, Sigma_0).
    Inputs: 
        y, m: array_like of shape (n) where n is number of observations
        X: array_like of shape (n, p) where p is number of parameters (covariates)
        beta_0: array_like of shape (p)
        Sigma_0_inv: array_like of shape (p, p)
        niter: number of sampled iterations to retune
        burnin: number of initial iterations to drop
        print_every: print progress every 'print_every' iterations
        retune: tune the proposal functions every 'retune' iterations
        verbose: print out accept rates in the previous retune period """
        
def bayes_logreg(m, y, X, beta_0, Sigma_0_inv, niter=10000, burnin=1000, print_every=1000,
                 retune=100, verbose=False):

    p = X.shape[1]  # dimension of x's
    samples = np.zeros([niter+burnin, p])
    samples[0, :] = np.zeros(p)

    # log-likelihood of the posterior 
    def log_post_pdf(beta):
        return np.dot(np.dot(X, beta), y) - 0.5*np.dot(beta-beta_0, np.dot(Sigma_0_inv,\
         beta-beta_0)) - np.dot(m, np.log(1+np.exp(np.dot(X, beta))))    
    
    stdevs = np.ones(p)
    accept_counts = np.zeros(p)

    # Metropolis with Gibbs
    for t in xrange(1, niter+burnin):
        beta_new = np.array(samples[t-1, :])
        for d in range(p):
            # generate new proposal with Gaussian
            beta_new_tmp = np.array(beta_new)
            beta_new_tmp[d] = np.random.normal(beta_new[d], stdevs[d])
            alpha = log_post_pdf(beta_new_tmp) - log_post_pdf(beta_new)
            if np.log(np.random.rand()) < alpha:   # accept or reject new proposal here
                beta_new[d] = beta_new_tmp[d]
                accept_counts[d] += 1
                
        samples[t, :] = beta_new

        # retune the standard deviation of the corresponding proposal to adjust accept rates    
        if t <= burnin and t % retune == 0:
            accept_rates = accept_counts/retune
            if verbose: 
                print 'Accept rates for the past ', p*retune, 'iterations are', accept_rates
            for i in range(p):
                if accept_rates[i] > 0.6:
                    stdevs[i] *= 1.25
                elif accept_rates[i] < 0.25:
                    stdevs[i] *= 0.75
            accept_counts = np.zeros(p)

        # print update info every 'print_every' iterations
        if (t+1) % print_every == 0 and verbose:
            print t+1, 'iterations just finished.'

    return samples[burnin:, :]
