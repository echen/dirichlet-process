'''
Code to calculate clusters using a Dirichlet Process
Gaussian mixture model. 

Requires scikit-learn:
  http://scikit-learn.org/stable/
'''

import numpy
from sklearn import mixture

FILENAME = "mcdonalds-normalized-data.tsv"

# Note: you'll have to remove the last "name" column in the file (or
# some other such thing), so that all the columns are numeric.
x = numpy.loadtxt(open(FILENAME, "rb"), delimiter = "\t", skiprows = 1)
dpgmm = mixture.DPGMM(n_components = 25)
dpgmm.fit(x)
clusters = dpgmm.predict(x)