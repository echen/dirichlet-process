import numpy
from sklearn import mixture

FILENAME = "mcdonalds-normalized-data.tsv"

x = numpy.loadtxt(open(FILENAME, "rb"), delimiter = "\t", skiprows = 1)
dpgmm = mixture.DPGMM(n_components = 25)
dpgmm.fit(x)
clusters = dpgmm.predict(x)