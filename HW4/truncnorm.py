import numpy as np

"""sampled from truncated normal distribution using approach by Robert 2009;
    use the straightforward rejection sampling if the interval is large enough"""
def truncnorm(mu, sigma, lo, hi, maxtries=500):
    x = None

    if hi == np.float32('inf') and np.isfinite(lo):
        #print 'left truncation'
        x = robert_one_sided(mu, sigma, lo, maxtries)

    elif np.isfinite(hi) and lo == -np.float32('inf'):
        #print 'right truncation'
        x = robert_one_sided(-mu, sigma, -hi, maxtries)
        if x:
            x = -x
    if not x:
        count = 0
        while count < maxtries and not x:
            tmp = np.random.normal(mu, sigma)
            if tmp >= lo and tmp <= hi:
                x = tmp
            count += 1

    if not x:
        print 'Failed to draw a sample from truncnorm(%f, %f, %f, %f); returning the mean' % \
        (mu, sigma, lo, hi)
        x = mu

    return x


def robert_one_sided(mu, sigma, lo, maxtries):
    x = None
    count = 0
    while count < maxtries and not x:
        mu_bar = (lo - mu)/sigma
        alpha = (mu_bar + np.sqrt(mu_bar**2.0+4.0))/2.0
        z = mu_bar + np.random.exponential(1.0/alpha)
        phi = np.exp(-(alpha-z)**2.0/2.0)
        if mu_bar >= alpha:
            phi *= np.exp(-(mu_bar-alpha)**2.0/2.0)
        if np.random.uniform() < phi:
            x = mu + sigma*z
        count += 1
   
    return x
