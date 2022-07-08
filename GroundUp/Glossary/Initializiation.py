"""
Interestingly initiliazing the weights of a neural network should not just be done at random, rather
you want to make sure that the eigenvalues are just 1 or close to it, so that you don't have exploding
or vanishing vectors when you show it the training data.

That's probably the first time I've seen functional analysis of transformations using eigenvalues and the resulting
implications. Really cool.
"""