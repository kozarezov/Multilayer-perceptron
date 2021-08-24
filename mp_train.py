#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import warnings

def get_data():
	try:
		data = np.genfromtxt("./data.csv", delimiter = ',')
		data_str = np.genfromtxt("./data.csv", delimiter = ',', dtype = np.str)
	except:
		print("Error")
		exit()
	x = data[:, 2:]
	tmp_y = np.zeros((len(data), 1))
	tmp_y[(data_str[:, 1] == 'M') == True] = 1
	y = np.zeros((len(tmp_y), int(max(tmp_y)) + 1))
	for i in range(0, int(max(tmp_y)) + 1):
		y[(tmp_y[:, 0] == i), i] = 1
	id_val = np.random.permutation(x.shape[0])[int(x.shape[0] * 0.7):]
	x_val = x[id_val, :]
	y_val = y[id_val, :]
	x = np.delete(x, id_val, axis = 0)
	y = np.delete(y, id_val, axis = 0)
	m = x.shape[0]
	m_val = x_val.shape[0]
	topology = [x.shape[1], 300, 300, y.shape[1]]
	theta = [np.random.randn(topology[1], topology[0] + 1)]
	for i in range (1, len(topology) - 1):
		theta.extend([np.random.randn(topology[i + 1], topology[i] + 1)])
	layers = len(topology) - 1
	epochs = 1300
	alpha = 0.001
	lmbd = 100
	momentum = 0.0003
	hist_loss = []; hist_val_loss = []; hist_cost = []
	return x, y, m, x_val, y_val, m_val, topology, layers, theta, epochs, alpha, lmbd, momentum, hist_loss, hist_val_loss, hist_cost

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_gradient(x):
	return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
	return np.exp(x) / sum(np.exp(x))

def print_status(a, y, m, x_val, y_val, m_val, theta, layers, epochs, i):
	## Loss
	true = 0; false = 0
	for j in range(0, len(a[layers])):
		if y[j, np.argmax(a[layers], axis = 1)[j]] == 1:
			true += 1.
		else:
			false += 1.
	loss = false / (true + false)
	## Cost
	cost = (-(np.dot(y.T, np.log(a[layers])) + \
		np.dot(1 - y.T, np.log(1 - a[layers]))) / m)[0][0]
	## Val_loss
	val_true = 0; val_false = 0
	a = [np.hstack((np.ones((m_val, 1)), x_val))]
	z = [np.dot(a[0], theta[0].T)]
	for j in range(0, layers - 1):
		a.extend([np.hstack((np.ones((len(z[j]), 1)), sigmoid(z[j])))])
		z.extend([np.dot(a[j + 1], theta[j + 1].T)])
	a.extend([np.apply_along_axis(softmax, axis = 1, arr = z[layers - 1])])
	for j in range(0, len(a[layers])):
		if y_val[j, np.argmax(a[layers], axis = 1)[j]] == 1:
			val_true += 1.
		else:
			val_false += 1.
	val_loss = val_false / (val_true + val_false)
	## Print
	print("epoch {:d}/{:d} - loss: {:f} - val_loss: {:f} - cost {:f}".format( \
		i, epochs, loss, val_loss, cost))
	return loss, val_loss, cost

def plot_data(epochs, hist_loss, hist_val_loss, hist_cost):
	plt.figure(1)
	plt_x = np.linspace(1, epochs, epochs)
	plt.plot(plt_x, hist_loss)
	plt.plot(plt_x, hist_val_loss)
	plt.legend(["Loss", "Val_loss"])
	plt.figure(2)
	plt.plot(plt_x, hist_cost)
	plt.legend(["Cost"])
	plt.show()

def main():
	x, y, m, x_val, y_val, m_val, topology, layers, theta, epochs, alpha, lmbd, momentum, hist_loss, hist_val_loss, hist_cost = get_data()
	for i in range(1, epochs + 1):
		## Feedforward
		a = [np.hstack((np.ones((m, 1)), x))]
		z = [np.dot(a[0], theta[0].T)]
		for j in range(0, layers - 1):
			a.extend([np.hstack((np.ones((len(z[j]), 1)), sigmoid(z[j])))])
			z.extend([np.dot(a[j + 1], theta[j + 1].T)])
		a.extend([np.apply_along_axis(softmax, axis = 1, arr = z[layers - 1])])
		## Backpropagation
		d = [a[layers] - y]
		for j in range(0, layers - 1):
			try:
				d.extend([np.dot(d[j], theta[-(j + 1)][:, 1:] - \
					momentum * theta_grad_prev[-(j + 1)][:, 1:]) * \
					sigmoid_gradient(z[-(j + 2)])])
			except:
				d.extend([np.dot(d[j], theta[-(j + 1)][:, 1:]) * \
					sigmoid_gradient(z[-(j + 2)])])
		## Gradient Update
		theta_grad = []
		for j in range(0, layers):
			theta_grad.extend([np.dot(d[-(j + 1)].T, a[j]) / m])
			theta_grad[j][:, 1:] += (lmbd * theta[j][:, 1:]) / m
			try:
				theta[j] -= momentum * theta_grad_prev[j] + alpha * theta_grad[j]
			except:
				theta[j] -= alpha * theta_grad[j]
		theta_grad_prev = theta_grad
		## Print Status
		loss, val_loss, cost = print_status(a, y, m, x_val, y_val, m_val, theta, layers, epochs, i)
		hist_loss.extend([loss])
		hist_val_loss.extend([val_loss])
		hist_cost.extend([cost])
	plot_data(epochs, hist_loss, hist_val_loss, hist_cost)
	np.savez("weights.npz", theta = theta, topology = topology)

if __name__ == "__main__":
	warnings.simplefilter("ignore")
	np.random.seed(21)
	main()
