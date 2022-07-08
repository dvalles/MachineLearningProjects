import numpy as np
import common.stats as stats

"""
Glossary:
Prior - a distribution that describes you're belief of some unknown variable. In the case below it'd be the 
mean of the data distribution. The idea is that you can never be sure what the real mean is so you model it with
a distribution. You can model anything you'd like, sky's the limit. Very powerful tool indeed.
Conjugate Prior - There are many common distributions that when modeling a parameter of that distribution you
can use a specific, closed form conjugate prior for the posterior update. For example, the conjugate prior 
of binomial is the beta distribution
Maximum A Posteriori - a numerical method for approximating the maximum likelihood function which is what
I was trying to figure out with the discrete case, but yeah you're basically trying to figure out what
parameter would make seeing the data you're seeing most likely

Thoughts:
Statistical modeling is much more flexible than I had originally realized. The idea that I can
model (create a distribution for) and parameter, outcome, etc of a model, is both awe inspiring
and scary. I believe I have a fairly firm grasp and could write a numerical method if necessary,
but I imagine there are still many concepts that would still prove unwieldy. For example, I
wouldn't know when it was necessary to use a gaussian process or why I might want to do that.

I find as well that the idea of modeling parameters of a function generator (like the coefficients
of a polynomial) is very exciting and certainly would be useful
"""

#---- standard counting way ----
print("")
print("Standard counting:")
mean, stddev = np.random.random()*20, np.random.random()*5
sample_count = 10000
sum = 0
all = []
for x in range(sample_count):
    sample = np.random.normal(mean, stddev) 
    sum += sample
    all.append(sample)
print("Real:", mean, stddev)
print("Calculated:", sum/sample_count, stats.StandardDeviation(all))
print("")


#---- bayesian with conjugate prior to estimate mean (normal normal model) ----
def MeanPosterior(sample, variance, variancePrior, meanPrior):
    """
    calculate the posterior mean
    """
    return ((variance*meanPrior) + (variancePrior*sample)) / (variance + variancePrior)

def VariancePosterior(variance, variancePrior):
    """
    calculate the posterior variance
    """
    return (variance*variancePrior)/(variance + variancePrior)
    
mean, stddev = np.random.random()*20, np.random.random()*5
variance = stddev ** 2
meanPrior, stddevPrior = 1, 1
variancePrior = stddevPrior ** 2
sample_count = 50
for x in range(sample_count):
    sample = np.random.normal(mean, stddev)
    meanPrior = MeanPosterior(sample, variance, variancePrior, meanPrior)
    variancePrior = VariancePosterior(variance, variancePrior)
stddevPrior = variancePrior ** .5

print("Real:", mean)
print("Calculated:", meanPrior, f'with confidence: {stddevPrior}')
print("")


#---- bayesian without conjugate prior ----
#the idea would be to maximize the likelihood by some numerical method, you could use gradient descent,
#random sampling, newtonian methods, etc. Whatever you like really