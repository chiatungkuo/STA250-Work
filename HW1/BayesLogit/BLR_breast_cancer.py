from pylab import *
from bayes_logreg import bayes_logreg

# read in and preprocess breast cancer input file 
infile = open('breast_cancer.txt', 'r')
lines = infile.readlines()
data = np.array(map(lambda x: x.rstrip('\r\n').split(), lines[1:-1]))
X = np.float_(data[:,:-1])
y = (data[:,-1] == '"M"')*1.0
m = np.ones(X.shape[0])
infile.close()

# fit a Baysian logistic regression 
p = X.shape[1]
beta_0 = np.zeros(p)
Sigma_0_inv = np.diag(0.001*np.ones(p))
samples = bayes_logreg(m, y, X, beta_0, Sigma_0_inv, niter=750000, burnin=50000)
#plot(samples)

# extract the posterior quantiles and write the result to file
beta_percentiles = np.zeros([99, p])
for i in range(99):
    for r in range(p):
        beta_percentiles[i, r] = np.percentile(samples[:, r], i+1)
np.savetxt('post_CI.csv', beta_percentiles, fmt='%.10f', delimiter=',')

# compute lag-1 and lag-p autocorrelation for each component
def autocorr(x, lag=1):
    y, z = x[lag:]-np.mean(x[lag:]), x[:-lag]-np.mean(x[:-lag])
    return np.sum(y*z) / np.sqrt(np.sum(y**2)*np.sum(z**2))

lag_1_autocorr = np.zeros(p)
lag_p_autocorr = np.zeros(p)
for i in range(p):
    lag_1_autocorr[i] = autocorr(samples[:, i])
    lag_p_autocorr[i] = autocorr(samples[:, i], lag=p)

print lag_1_autocorr, lag_p_autocorr

# sample M predictive datasets 
M = 100
y_pred = np.zeros([y.shape[0], M])
indices = np.random.randint(0, samples.shape[0], size=M)
for i in range(M):
    beta = samples[indices[i], :]
    u = np.dot(X, beta)
    y_pred[:, i] = np.random.binomial(1, np.exp(u) / (1+np.exp(u)))

# perform posterior predictive checks for the means
print np.mean(y), np.mean(np.mean(y_pred, 0)), np.std(np.mean(y_pred, 0))
