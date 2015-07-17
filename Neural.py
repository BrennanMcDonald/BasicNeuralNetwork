import numpy as np
import itertools

def sig(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

X = np.array([
	[0,0,1,0,0],
	[0,1,1,0,0],
	[1,0,1,0,0],
	[0,0,0,0,1],
	[0,1,0,0,0],
	[1,0,0,0,0],
	[0,1,1,0,0]])

Y = np.array([[1],[0],[0],[1],[1],[1],[0]])

np.random.seed(1)

#weights_0 = np.zeros(shape=(5,6))
#weights_1 = np.zeros(shape=(6,6))
#weights_2 = np.zeros(shape=(6,6))
#weights_3 = np.zeros(shape=(6,1))

weights_0 = 2*np.random.random((5,6)) - 1
weights_1 = 2*np.random.random((6,6)) - 1
weights_2 = 2*np.random.random((6,6)) - 1
weights_3 = 2*np.random.random((6,1)) - 1

prevFail = 0
prevSuccess = 0

for iter in range(10000000000):

	input_layer = X #l0
	hidden_layer = sig(np.dot(input_layer, weights_0)) #l1
	hidden_layer_2 = sig(np.dot(hidden_layer, weights_1)) #l2
	hidden_layer_3 = sig(np.dot(hidden_layer_2, weights_2)) #l2
	output_layer = sig(np.dot(hidden_layer_3, weights_3)) #l3

	l4_error = Y - output_layer
	l4_delta = l4_error * sig(output_layer,True)
	l3_error = l4_delta.dot(weights_3.T)
	l3_delta = l3_error * sig(hidden_layer_3,True)
	l2_error = l3_delta.dot(weights_2.T)
	l2_delta = l2_error * sig(hidden_layer_2,True)
	l1_error = l2_delta.dot(weights_1.T)
	l1_delta = l1_error * sig(hidden_layer,True)

	weights_3 += hidden_layer_3.T.dot(l4_delta)
	weights_2 += hidden_layer_2.T.dot(l3_delta)
	weights_1 += hidden_layer.T.dot(l2_delta)
	weights_0 += input_layer.T.dot(l1_delta)

	if (iter % 10000 == 0):
		failed = 0
		success = 0

		poss = [[0,0,0,0,0],[1,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1]]
		for p in poss:
			for i in itertools.permutations(p):
				input_layer = i
				hidden_layer = sig(np.dot(input_layer, weights_0)) #l1
				hidden_layer_2 = sig(np.dot(hidden_layer, weights_1)) #l2
				hidden_layer_3 = sig(np.dot(hidden_layer_2, weights_2)) #l2
				output_layer = sig(np.dot(hidden_layer_3, weights_3)) #l3
				if ((str(sum(i)) != str(output_layer.round())[2]) and str(output_layer.round())[2] == "1"):
					failed += 1
				else :
					success += 1

		if (prevSuccess != success):
			print("Faield: " + str(failed))
			print("Successful: " + str(success))
		prevFail = failed
		prevSuccess = success
