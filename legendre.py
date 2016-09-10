import numpy
from numpy.polynomial import legendre as L
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import Polynomial
from numpy.random import randn
from scipy import stats
import timeit
import matplotlib.pyplot as plt

def normlegInteg(coefs):
	lOrig = L.Legendre(coefs) 
	l2integ = (lOrig ** 2).integ()
	d = ((l2integ(1)-l2integ(-1))/2)**0.5
	return lOrig/d


def normlegExpect(coefs):
	lOrig = L.Legendre(coefs) 
	d = (stats.uniform.expect(lOrig**2,loc=-1, scale=2))**0.5
	return lOrig/d

def genlegpoly(degree):
	return  normlegInteg(randn(degree+1))

def gendataset(legpoly,variance,n):
	x = numpy.random.uniform(-1.0,1.0,n)
	y = legpoly(x) + (variance * randn(n))
	return (x,y)



leg = genlegpoly(50)
#print(stats.uniform.expect(leg**2,loc=-1, scale=2))

leg_pts = leg.linspace(1000)
plt.plot(leg_pts[0],leg_pts[1])

ds = gendataset(leg,0.1,100)
plt.plot(ds[0],ds[1],'ro')

#h2 = polyfit(ds[0],ds[1],2)
h2 = Polynomial.fit(ds[0],ds[1],2,[-1,1])
h2_pts = h2.linspace(1000)
plt.plot(h2_pts[0],h2_pts[1],'g')

h10 = Polynomial.fit(ds[0],ds[1],10,[-1,1])
h10_pts = h10.linspace(1000)
plt.plot(h10_pts[0],h10_pts[1],'r')


print h2

x0, x1, y0, y1 = plt.axis()
plt.axis((-2.0,2.0,y0,y1))
plt.show()
# def normlegInteg_test():
#     normlegInteg(range(100))

# def normlegExpect_test():
#     normlegExpect(range(100))

# print(timeit.timeit(normlegInteg_test,number=1000))
# print(timeit.timeit(normlegExpect_test,number=1000))

