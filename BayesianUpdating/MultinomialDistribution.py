import common.probability as probs
import numpy as np
import common.linearalgebra as la
import common.stats as stats

"""
Seeing if I can derive the update function for approximating an
unknown multinomial distribution myself before moving on to higher
dimensional and then continuous cases

Strategy:
Add up outcomes and find the percentages

You could also solve for the distribution that is most likely to give you the seen sequence,
create and equation and solve for which variables give you the max likelihood while following
the add up to 1 constraint, but that just ends up being the same as above with more steps
"""

#no prior, just lots of samples strat
num_samples = 100000
tri = probs.RandomTrinomialDistribution()
counts = {0: 0, 1: 0, 2: 0}
for x in range(num_samples):
    counts[probs.SampleDiscreteDistribution(tri, [0,1,2])] += 1
approx = dict(map(lambda tup: (tup[0], tup[1]/num_samples), counts.items()))
diff = la.Subtract(tri, approx)
errors = list(map(lambda x : abs(x), diff))
error = stats.Mean(errors)
print(f'Approximation error: {round(error, 5)}')


#bayesian approach
#given a search online, looks like this is perhaps an unusual thing to do
#but you can use a dirichlet prior if you'd like