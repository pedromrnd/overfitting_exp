from numpy.polynomial import legendre as L
from numpy.random import randn
from scipy import stats
from scipy import stats
import timeit
import Tkinter
import matplotlib.pyplot as plt
import random

def normlegInteg(coefs):
	lOrig = L.Legendre(coefs) 
	l2integ = (lOrig ** 2).integ()
	d = ((l2integ(1)-l2integ(-1))/2)**0.5
	#lnorm = lOrig/d
	# print(lOrig)
	# print(lnorm)
	# print(stats.uniform.expect(lOrig**2,loc=-1, scale=2))
	# print(stats.uniform.expect(lnorm**2,loc=-1, scale=2))
	return lOrig/d


def normlegExpect(coefs):
	lOrig = L.Legendre(coefs) 
	d = (stats.uniform.expect(lOrig**2,loc=-1, scale=2))**0.5
	return lOrig/d

leg = normlegInteg(randn(3))
print(stats.uniform.expect(leg**2,loc=-1, scale=2))

pts = leg.linspace(1000)
plt.plot(pts[0],pts[1])
x0, x1, y0, y1 = plt.axis()
plt.axis((-2.0,2.0,y0,y1))
plt.show()
# def normlegInteg_test():
#     normlegInteg(range(100))

# def normlegExpect_test():
#     normlegExpect(range(100))

# print(timeit.timeit(normlegInteg_test,number=1000))
# print(timeit.timeit(normlegExpect_test,number=1000))

