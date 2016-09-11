
import numpy as np
from numpy.polynomial import legendre as L
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import Polynomial
from numpy.random import randn
from scipy import stats
import timeit
import matplotlib.pyplot as plt

class LengedreExperiment:
	def __init__(self, qf, variance, n):
		self.qf = qf
		self.variance = variance
		self.n = n
		self.experiment()

	def normlegInteg(self, coefs):
		lOrig = L.Legendre(coefs) 
		l2integ = (lOrig ** 2).integ()
		d = ((l2integ(1)-l2integ(-1))/2)**0.5
		return lOrig/d


	def normlegExpect(self, coefs):
		lOrig = L.Legendre(coefs) 
		d = (stats.uniform.expect(lOrig**2,loc=-1, scale=2))**0.5
		return lOrig/d

	def genlegpoly(self):
		return  self.normlegInteg(randn(self.qf+1))

	def gendataset(self):
		x = np.random.uniform(-1.0,1.0,self.n)
		y = self.leg(x) + (self.variance * randn(self.n))
		return (x,y)

	def mse(self, prediction, target):
		diffinteg = (((prediction - Polynomial.cast(target))**2)).integ()
		return np.sqrt((diffinteg(1) - diffinteg(-1))/2)

	def mse2(self, prediction, target, sample_size):
		sumac = 0.0
		for x in np.random.uniform(-1.0,1.0,sample_size):
			sumac += (prediction(x) - target(x))**2
		return (sumac/sample_size)**0.5

	def experiment(self):
		self.leg = self.genlegpoly()
		self.dataset = self.gendataset()
		self.g2 = Polynomial.fit(self.dataset[0],self.dataset[1],2,[-1,1])
		self.g10 = Polynomial.fit(self.dataset[0],self.dataset[1],10,[-1,1])
		self.eoutg2 = self.mse(self.g2, self.leg)
		self.eoutg10 = self.mse(self.g10, self.leg)

	def printstats(self):
		print("Eout(g2):  %f"%(self.eoutg2))
		print("Eout(g10): %f"%(self.eoutg10))

	def plot(self):
		fn_points = 10000
		leg_pts = self.leg.linspace(fn_points)
		g2_pts = self.g2.linspace(fn_points)
		g10_pts = self.g10.linspace(fn_points)
		plt.plot(leg_pts[0],leg_pts[1])
		plt.plot(self.dataset[0],self.dataset[1],'bo')
		plt.plot(g2_pts[0], g2_pts[1],'g')
		plt.plot(g10_pts[0],g10_pts[1],'r')
		x0, x1, y0, y1 = plt.axis()
		plt.axis((-2.0,2.0,y0,y1))
		plt.show()


exp = LengedreExperiment(5,0.1,200)

#exp.stats()
#exp.plot()
# leg = genlegpoly(50)
# #print(stats.uniform.expect(leg**2,loc=-1, scale=2))

# leg_pts = leg.linspace(1000)
# plt.plot(leg_pts[0],leg_pts[1])

# ds = gendataset(leg,0.1,100)
# plt.plot(ds[0],ds[1],'ro')

# #h2 = polyfit(ds[0],ds[1],2)
# h2 = Polynomial.fit(ds[0],ds[1],2,[-1,1])
# h2_pts = h2.linspace(1000)
# plt.plot(h2_pts[0],h2_pts[1],'g')

# h10 = Polynomial.fit(ds[0],ds[1],10,[-1,1])
# h10_pts = h10.linspace(1000)
# plt.plot(h10_pts[0],h10_pts[1],'r')


# print h2

# x0, x1, y0, y1 = plt.axis()
# plt.axis((-2.0,2.0,y0,y1))
# plt.show()
# # def normlegInteg_test():
#     normlegInteg(range(100))

# def normlegExpect_test():
#     normlegExpect(range(100))

# print(timeit.timeit(normlegInteg_test,number=1000))
# print(timeit.timeit(normlegExpect_test,number=1000))

