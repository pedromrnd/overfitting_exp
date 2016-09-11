import json
import sys
import matplotlib.pyplot as plt
import numpy as np

file = sys.argv[1]

def plot_stochastic_noise(file, nlbound=60, nubound=130, vlbound=0.0, vubound=2.5, vstep=0.05, qf=20):
	nsize = nubound - nlbound + 1
	vsize = int(round((vubound - vlbound)/vstep)) + 1
	print vsize
	matrix = create_matrix(vsize, nsize)
	with open(file, 'r') as f:
		for line in f:
			exp = json.loads(line)
			if exp['n'] < nlbound or exp['n'] > nubound:
				continue
			if exp['variance'] < vlbound or exp['variance'] > vubound:
				continue
			n_idx = exp['n'] - nlbound
			v_idx = int(round((exp['variance'] - vlbound )/ vstep))
			#print(exp['n'])
			#print(exp['variance'])
			val = (exp['eouth10']) - exp['eouth2']
			val = min(max(-0.2, val),0.2)
			matrix[v_idx][n_idx] = val
		for i, row in enumerate(matrix):
			for j,element in enumerate(row):
				if element is None:
					#pass
					print(i)
					print(j)
	#plt.scatter([range(140-1)],np.linspace(0.0, 2.5, num=51),matrix)
	X, Y = np.meshgrid(range(nlbound, nubound+1),np.linspace(vlbound, vubound, num=vsize))
	# print X
	# print Y
	# X, Y = np.meshgrid(range(2), range(2))
	# print X
	# print Y
	# matrix = [
	# 		  [0.1,0.2,0.3],
	# 		  [0.1,0.2,0.3]
	# 		 ]
	plt.pcolor(X,Y, matrix, vmin=-0.2, vmax=0.2)
	plt.colorbar()
	plt.show()


def create_matrix(i, j):
	matrix=[]
	for ii in range(i):
		matrix.append([])
		for jj in range(j):
			matrix[ii].append(None)
	return matrix

plot_stochastic_noise(file)