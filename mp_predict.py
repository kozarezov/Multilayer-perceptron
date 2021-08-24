#! /usr/bin/env python3
import numpy as np
import warnings
import sys

def get_data():
	try:
		data = np.load("./weights.npz", allow_pickle=True)
	except:
		print("Weights Error")
		exit()
	theta = data["theta"]
	topology = data["topology"]
	try:
		data = np.genfromtxt(sys.argv[1], delimiter = ',')
		data_str = np.genfromtxt(sys.argv[1], delimiter = ',', dtype = np.str)
	except:
		print("Error")
		print("Usage : python nn_predict.python [dataset]")
		exit()
	x = data[:, 2:]
	tmp_y = np.zeros((len(data), 1))
	tmp_y[(data_str[:, 1] == 'M') == True] = 1
	y = np.zeros((len(tmp_y), int(max(tmp_y)) + 1))
	for i in range(0, int(max(tmp_y)) + 1):
		y[(tmp_y[:, 0] == i), i] = 1
	m = x.shape[0]
	layers = len(topology) - 1
	return x, y, m, topology, layers, theta

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def softmax(x):
	return np.exp(x) / sum(np.exp(x))

def main():
	x, y, m, topology, layers, theta = get_data()
	a = [np.hstack((np.ones((m, 1)), x))]
	z = [np.dot(a[0], theta[0].T)]
	for j in range(0, layers - 1):
		a.extend([np.hstack((np.ones((len(z[j]), 1)), sigmoid(z[j])))])
		z.extend([np.dot(a[j + 1], theta[j + 1].T)])
	a.extend([np.apply_along_axis(softmax, axis = 1, arr = z[layers - 1])])
	cost = (-(np.dot(y.T, np.log(a[layers])) + \
		np.dot(1 - y.T, np.log(1 - a[layers]))) / m)[0][0]
	print("Cross entropy loss : {:f}".format(cost))
	np.savetxt("prediction.csv", a[layers], delimiter = ',', fmt = "%.6f")

if __name__ == "__main__":
	warnings.simplefilter("ignore")
	np.random.seed(1)
	main()
